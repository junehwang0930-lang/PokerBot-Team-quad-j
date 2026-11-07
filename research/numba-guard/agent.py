"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

Team: quad-j. Bot: jish_new.
Strategy: precomputed preflop equity, compiled Monte Carlo evaluation,
opponent range inference, learned interpretation of checks, and half-local
adaptation. See BOT_NOTES.md.

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# Tournament identity
PARTICIPANT = "Team-quad-j"
AGENT_NAME = "AdaptiveJishGuard"


# ===== YOUR CODE HERE =====

# The strategy and its numerical helpers live inside the editable section.
# Numba compiles these loops during startup; matches need no network or writes.
import math
import random
import time
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
from numba import njit
from evaluator import _score_ranks

_RANK_POWER = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
_SUIT_BIT = np.array([1 << (c & 3) for c in range(36)], dtype=np.int32)
_BINOM = np.zeros((36, 6), dtype=np.int32)
for _c in range(36):
    for _k in range(6):
        _BINOM[_c, _k] = math.comb(_c, _k) if _c >= _k else 0
_SCORE = np.zeros((2, 5 ** 9), dtype=np.int32)
for _ranks in combinations_with_replacement(range(9), 5):
    if max(_ranks.count(r) for r in _ranks) > 4:
        continue
    _key = sum(5 ** r for r in _ranks)
    _SCORE[0, _key] = _score_ranks(tuple(reversed(_ranks)), False)
    _SCORE[1, _key] = _score_ranks(tuple(reversed(_ranks)), True)


@njit
def _index(hole):
    ordered = np.sort(hole)
    idx = 0
    for i in range(5):
        idx += _BINOM[ordered[i], i + 1]
    return idx


@njit
def _pairs(hole):
    ranks = np.empty(10, dtype=np.int32)
    suits = np.empty(10, dtype=np.int32)
    n = 0
    for i in range(5):
        for j in range(i + 1, 5):
            ranks[n] = _RANK_POWER[hole[i]] + _RANK_POWER[hole[j]]
            suits[n] = _SUIT_BIT[hole[i]] & _SUIT_BIT[hole[j]]
            n += 1
    return ranks, suits


@njit
def _score_pairs(ranks, suits, board, length, table):
    best = 0
    for i in range(length - 2):
        for j in range(i + 1, length - 1):
            for k in range(j + 1, length):
                br = _RANK_POWER[board[i]] + _RANK_POWER[board[j]] + _RANK_POWER[board[k]]
                bs = _SUIT_BIT[board[i]] & _SUIT_BIT[board[j]] & _SUIT_BIT[board[k]]
                for p in range(10):
                    value = table[int((bs & suits[p]) != 0), br + ranks[p]]
                    if value > best:
                        best = value
    return best


@njit
def _fast_score(hole, board, table):
    ranks, suits = _pairs(hole)
    return _score_pairs(ranks, suits, board, len(board), table)


@njit
def _random_card(state):
    # Per-call PRNG state, never a global stream carried between halves.
    state ^= state << np.uint64(13)
    state ^= state >> np.uint64(7)
    state ^= state << np.uint64(17)
    return state, int(state % np.uint64(36))


@njit
def _sample(hole, board, count, runouts, seed, table, preflop):
    visible = len(board)
    known = np.uint64(0)
    for c in hole:
        known |= np.uint64(1) << np.uint64(c)
    for c in board:
        known |= np.uint64(1) << np.uint64(c)
    state = np.uint64(seed)
    hr, hs = _pairs(hole)
    opponents = np.empty((count, 5), dtype=np.int32)
    scores = np.zeros((count, 4), dtype=np.float64)
    outcomes = np.empty(count, dtype=np.float64)
    future = np.empty(5, dtype=np.int32)
    for i in range(visible):
        future[i] = board[i]
    for n in range(count):
        used = known
        for i in range(5):
            while True:
                state, card = _random_card(state)
                bit = np.uint64(1) << np.uint64(card)
                if not used & bit:
                    break
            opponents[n, i] = card
            used |= bit
        other = opponents[n]
        ranks, suits = _pairs(other)
        scores[n, 0] = preflop[_index(other)]
        for street in range(1, 4):
            length = street + 2
            if length <= visible:
                scores[n, street] = _score_pairs(ranks, suits, future, length, table)
        won = 0.0
        for run in range(runouts):
            run_used = used
            for i in range(visible, 5):
                while True:
                    state, card = _random_card(state)
                    bit = np.uint64(1) << np.uint64(card)
                    if not run_used & bit:
                        break
                future[i] = card
                run_used |= bit
            mine = _score_pairs(hr, hs, future, 5, table)
            theirs = _score_pairs(ranks, suits, future, 5, table)
            won += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        outcomes[n] = won / runouts
    return opponents, scores, outcomes


_DATA = Path(__file__).resolve().parent / 'data' / 'preflop.npy'
if _DATA.exists():
    _PREFLOP = np.load(_DATA, allow_pickle=False)
elif os.environ.get('QUADJ_PRECOMPUTE') == '1':
    _PREFLOP = np.full(376992, 0.5, dtype=np.float32)
else:
    raise FileNotFoundError('Missing data/preflop.npy; include the data directory with agent.py')
if _PREFLOP.shape != (376992,) or _PREFLOP.dtype != np.float32 or not np.isfinite(_PREFLOP).all():
    raise ValueError('Invalid preflop equity table')
# Force compilation before serving requests. No compilation is charged to a half.
_sample(np.array([0, 5, 10, 15, 20], dtype=np.int32),
        np.array([24, 29, 34], dtype=np.int32), 1, 1, np.uint64(1), _SCORE, _PREFLOP)
_fast_score(np.array([0, 5, 10, 15, 20], dtype=np.int32),
            np.array([24, 29, 34], dtype=np.int32), _SCORE)


def new_half(config: AgentConfig, memory: dict) -> None:
    """Reset all learning, random state, caches, and the compute allowance."""
    memory.clear()
    memory.update(hands_played=0, config=config, rng=random.Random(19470219),
                  counts=np.zeros((4, 2, 3), dtype=float), check_evidence=np.zeros((4,2)),
                  hand=-1, seen=0, cache=None, last_obs=None, last_action=None,
                  compute_seconds=0.0, decisions=0, fallback_count=0,
                  strength_sum=0.0, strength_count=0, net=0)


def _observe(obs, memory):
    if memory['hand'] != obs.hand_index:
        memory.update(hand=obs.hand_index, seen=0, cache=None)
    # Replay public bets to distinguish a check from a call and an opening
    # bet from a re-raise. Never consult the deal or future board cards.
    config = memory['config']
    bets = [config.small_blind if obs.is_button else config.big_blind,
            config.big_blind if obs.is_button else config.small_blind]
    street, raises = 0, 0
    total = bets.copy()
    for i, entry in enumerate(obs.action_history):
        st = int(entry.street)
        if st != street:
            bets, street, raises = [0, 0], st, 0
        seat = 0 if entry.is_me else 1
        facing = int(bets[1 - seat] > bets[seat])
        can_raise = (raises < memory['config'].raise_cap and
                     total[1 - seat] < memory['config'].starting_stack and
                     memory['config'].starting_stack - total[seat] > max(0, bets[1-seat] - bets[seat]))
        if not entry.is_me and i >= memory['seen'] and can_raise:
            kind = 0 if entry.action == Action.FOLD else (2 if entry.action in (Action.RAISE_HALF, Action.RAISE_POT) else 1)
            memory['counts'][st, facing, kind] += 1
        if entry.action in (Action.RAISE_HALF, Action.RAISE_POT):
            raises += 1
        bets[seat] = entry.street_total
        total[seat] += entry.amount
    memory['seen'] = len(obs.action_history)


def _rates(memory, street, facing):
    counts = memory['counts']
    # Share information between postflop streets, with modest street-specific weight.
    if street == 0:
        c = counts[0, facing] + 0.18 * counts[1:, facing].sum(axis=0)
    else:
        c = counts[1:, facing].sum(axis=0) + counts[street, facing]
    prior = np.array([2.4, 3.6, 2.0]) if facing else np.array([0., 5., 3.])
    p = (c + prior) / (c.sum() + prior.sum())
    return p


def _percentile(values):
    ordered = np.sort(values)
    return (np.searchsorted(ordered, values, 'left') + np.searchsorted(ordered, values, 'right')) / (2.0 * len(values))


def _range_weights(obs, memory, scores):
    weights = np.ones(len(scores))
    pct = [_percentile(scores[:, st]) for st in range(int(obs.street) + 1)]
    config = memory['config']
    bets = [config.small_blind if obs.is_button else config.big_blind,
            config.big_blind if obs.is_button else config.small_blind]
    street = 0
    for entry in obs.action_history:
        st = int(entry.street)
        if st != street:
            bets, street = [0, 0], st
        seat = 0 if entry.is_me else 1
        facing = int(bets[1 - seat] > bets[seat])
        if not entry.is_me:
            p = _rates(memory, st, facing)
            aggressive = p[2]
            # Frequent aggression carries less information about card strength.
            selective = max(0.0, min(1.0, (0.78 - aggressive) / 0.50))
            if memory['strength_count'] >= 10:
                mean = (memory['strength_sum'] + 6.0) / (memory['strength_count'] + 8.0)
                selective *= max(0.08, min(1.0, (mean - 0.48) / 0.23))
            strength = pct[st]
            if entry.action in (Action.RAISE_HALF, Action.RAISE_POT):
                threshold = min(0.93, 1.0 - aggressive + (0.06 if facing else 0.0))
                likelihood = 0.09 + 0.91 / (1.0 + np.exp(-12.0 * (strength - threshold)))
                weights *= (1.0 - selective) + selective * likelihood
            elif facing:
                threshold = min(0.65, p[0])
                likelihood = 0.15 + 0.85 / (1.0 + np.exp(-10.0 * (strength - threshold)))
                weights *= (1.0 - selective) + selective * likelihood
            else:
                bet_likelihood = 0.09 + 0.91 / (1.0 + np.exp(-12.0 * (strength - (1.0-aggressive))))
                old_check = 1.0 - selective * .55 * strength ** 4
                new_check = 1.0 - selective * bet_likelihood
                information = _check_information(memory,st) if st else .5
                weights *= (1-information)*old_check + information*new_check
        bets[seat] = entry.street_total
    weights = np.maximum(weights, 1e-5)
    return weights / weights.sum(), pct[-1]


def _simulation(obs, memory):
    key = (obs.hand_index, obs.hole_cards, obs.board)
    cached = memory['cache']
    if cached is not None and cached['key'] == key:
        return cached
    # Keep substantial time for HTTP overhead and slower tournament hardware.
    hands = max(1, memory['config'].num_hands)
    allowance = memory['config'].time_bank_seconds * 0.40 / hands
    spent_per_hand = memory['compute_seconds'] / max(1, obs.hand_index + 1)
    scale = min(1.0, allowance / max(allowance, spent_per_hand))
    n = max(96, int((768 if obs.street != Street.RIVER else 1536) * scale))
    hole = np.asarray(obs.hole_cards, dtype=np.int32)
    board = np.asarray(obs.board, dtype=np.int32)
    other, scores, outcomes = _sample(hole, board, n, 3 if len(board) < 5 else 1,
                                      np.uint64(memory['rng'].getrandbits(63) | 1), _SCORE, _PREFLOP)
    cached = dict(key=key, opponents=other, scores=scores, outcomes=outcomes)
    memory['cache'] = cached
    return cached


def _decide(obs, memory):
    _observe(obs, memory)
    st = int(obs.street)
    free = obs.to_call == 0
    raising = Action.RAISE_POT in obs.legal_actions
    p_free = _rates(memory, st, 0)
    p_facing = _rates(memory, st, 1)
    if st == 0:
        equity = float(_PREFLOP[_index(np.asarray(obs.hole_cards, dtype=np.int32))])
        # Preflop equities are precomputed from random legal deals. Raise
        # histories then shift the estimate toward the opponent's stronger range.
        opp_raises = sum(not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT) for a in obs.action_history)
        selectivity = max(0.0, (0.78 - p_facing[2]) / 0.55)
        adjusted = equity - selectivity * min(0.14, 0.035 * opp_raises)
        if raising:
            if obs.raises_this_street == 0:
                if (obs.is_button and equity >= 0.48) or equity >= 0.56:
                    return Action.RAISE_POT
                # Steal against observed over-folding without risking a large pot.
                if p_facing[0] > 0.63:
                    return Action.RAISE_HALF
            elif adjusted > (0.57 if p_facing[2] < 0.65 else 0.525):
                return Action.RAISE_POT
        if free:
            return Action.CHECK_CALL
        cost = min(obs.to_call, obs.my_stack)
        odds = cost / (obs.pot + cost)
        # Against relentless raisers, marginal preflop calls commit the stack
        # over subsequent raises, so require near break-even all-in equity.
        if p_facing[2] > 0.78 and obs.opp_stack > 0:
            threshold = min(0.495, (obs.my_stack / (obs.pot + 2 * obs.my_stack - obs.to_call)))
        else:
            spr = min(obs.my_stack - cost, obs.opp_stack) / max(1, obs.pot + cost)
            threshold = odds + (0.10 if spr > 2 else 0.045 if spr > 0.5 else 0.0)
        return Action.CHECK_CALL if adjusted >= threshold else Action.FOLD
    simulation = _simulation(obs, memory)
    weights, strength = _range_weights(obs, memory, simulation['scores'])
    outcomes = simulation['outcomes']
    equity = float(np.dot(weights, outcomes))
    cost = min(obs.to_call, obs.my_stack)
    odds = cost / max(1, obs.pot + cost)
    spr = max(0, min(obs.my_stack - cost, obs.opp_stack)) / max(1, obs.pot + cost)
    # Future betting reduces the value of weak draws out of position.
    reserve = (0.045 if not obs.is_button else 0.025) * min(1.0, spr) if st < 3 else 0.0
    if not free and equity < odds + reserve:
        return Action.FOLD
    if not raising:
        return Action.CHECK_CALL
    # Approximate the continuing range for each available size. Large bets
    # must earn their chips against hands that actually call, not folded hands.
    base_ev = equity * (obs.pot + cost) - cost
    best_action, best_ev = Action.CHECK_CALL, base_ev
    for action in (Action.RAISE_HALF, Action.RAISE_POT):
        if action not in obs.legal_actions:
            continue
        invest = obs.raise_to[action] - obs.my_bet
        reply = min(obs.opp_stack, max(0, obs.raise_to[action] - obs.opp_bet))
        risk_ratio = reply / max(1, obs.pot + invest + reply)
        fold_rate = p_facing[0]
        if not free:
            fold_rate *= 0.55
        fold_rate *= 0.85 + risk_ratio
        # Fit continuation strength to observed folds. Calling stations and
        # maniacs retain almost their entire range.
        selectivity = max(0.0, min(1.0, (0.80 - p_facing[2]) / 0.55))
        continue_p = 1.0 - fold_rate * ((1.0 - selectivity) + selectivity * 2.0 * (1.0 - strength))
        continue_p = np.clip(continue_p, 0.03, 1.0)
        continuing = weights * continue_p
        call_prob = float(continuing.sum())
        eq_called = float(np.dot(continuing, outcomes) / call_prob)
        ev = (1 - call_prob) * obs.pot + call_prob * (eq_called * (obs.pot + invest + reply) - invest)
        # A one-step estimate overvalues bluffs that can be re-raised and
        # medium hands with several streets still to play.
        if st < 3:
            ev -= call_prob * invest * max(0.0, 0.62 - eq_called) * (0.70 if spr > 1 else 0.25)
        margin = max(0.35, obs.pot * (0.035 if st < 3 else 0.018))
        if ev > best_ev + margin:
            best_action, best_ev = action, ev
    return best_action


def choose_action(obs: Observation, memory: dict) -> Action:
    """Return a legal action using equity, public betting, and half-local learning."""
    start = time.perf_counter()
    try:
        action = _decide(obs, memory)
    except (ArithmeticError, ValueError, IndexError, KeyError):
        # A legal conservative fallback, counted so tests can detect its use.
        memory['fallback_count'] = memory.get('fallback_count', 0) + 1
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    memory['last_obs'], memory['last_action'] = obs, action
    memory['decisions'] += 1
    memory['compute_seconds'] += time.perf_counter() - start
    return action if action in obs.legal_actions else Action.CHECK_CALL


def _learn_checks(result, memory):
    """Learn from revealed cards only; shrink sparse evidence toward the prior."""
    obs, cache = memory.get('last_obs'), memory.get('cache')
    if not result.opp_hole or obs is None or cache is None or obs.hand_index != result.hand_index or cache['key'][0] != result.hand_index:
        return
    other = np.asarray(result.opp_hole,dtype=np.int32)
    cfg = memory['config']
    bets = [cfg.small_blind if obs.is_button else cfg.big_blind,
            cfg.big_blind if obs.is_button else cfg.small_blind]
    street = 0
    learned = set()
    for entry in obs.action_history:
        st = int(entry.street)
        if st != street:
            bets,street = [0,0],st
        seat = 0 if entry.is_me else 1
        if st and len(cache['key'][2]) >= st + 2 and not entry.is_me and entry.action == Action.CHECK_CALL and bets[0] <= bets[1] and st not in learned:
            learned.add(st)
            actual = _fast_score(other,np.asarray(obs.board[:st+2],dtype=np.int32),_SCORE)
            pool = cache['scores'][:,st]
            percentile = float(np.mean(pool < actual)+.5*np.mean(pool == actual))
            memory['check_evidence'][st,0] += percentile
            memory['check_evidence'][st,1] += 1
        bets[seat] = entry.street_total


def _check_information(memory, street):
    # Showdown samples are selected by the betting. This is cautious evidence
    # of slow-playing, not an unbiased estimate of the entire checking range.
    evidence = memory['check_evidence'][1:].sum(axis=0) + memory['check_evidence'][street]
    mean = (evidence[0]+12*.50)/(evidence[1]+12)
    return max(0.,min(1.,(.54-mean)/.12))


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Learn terminal responses, which do not appear in the next observation."""
    _learn_checks(result,memory)
    memory['hands_played'] += 1
    memory['net'] += result.my_net
    obs, action = memory.get('last_obs'), memory.get('last_action')
    if obs is None or obs.hand_index != result.hand_index:
        return
    if (action in (Action.RAISE_HALF, Action.RAISE_POT)
            and obs.raises_this_street + 1 < memory['config'].raise_cap
            and obs.my_stack > obs.raise_to[action] - obs.my_bet
            and obs.opp_stack > obs.raise_to[action] - obs.opp_bet):
        if not result.went_to_showdown and result.winner == 'me':
            memory['counts'][int(obs.street), 1, 0] += 1
        elif result.went_to_showdown:
            memory['counts'][int(obs.street), 1, 1] += 1
    cached = memory.get('cache')
    if result.opp_hole and cached is not None and cached['key'][0] == result.hand_index:
        other = np.asarray(result.opp_hole, dtype=np.int32)
        for st in set(int(a.street) for a in obs.action_history if not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT)):
            if st and len(cached['key'][2]) < st + 2:
                continue
            if st == 0:
                value = float(_PREFLOP[_index(other)])
            else:
                value = _fast_score(other, np.asarray(obs.board[:st+2], dtype=np.int32), _SCORE)
            pool = cached['scores'][:, st]
            memory['strength_sum'] += float(np.mean(pool < value) + 0.5 * np.mean(pool == value))
            memory['strength_count'] += 1


# Reuse the frozen champion's already validated aggression guard. The embedded
# policy is fixed game knowledge, and all of its observations reset each half.
import base64 as _guard_base64
import zlib as _guard_zlib
import types as _guard_types
_CHAMP = _guard_types.ModuleType("embedded_champion")
_CHAMP.__file__ = __file__
exec(_guard_zlib.decompress(_guard_base64.b85decode('c-rlKYjYdNneKP~idnnr2#`Q(Zao*qbRF52H_AGeN_J9}VzF2t35tk7fB`_sN;d!fJkNXSnE^>zPIC6_*0Jo8fCoK&eeeBt&u5=~wl{q<TW$|-J|z3epI-jqb#k+MJ6k7btL;JZ<JEkVtd_Ina=w@)_&iO{SJyX-*>;w^pPeO}+4|jVeXzIp`gWPjw}*QP{@i@nUagi%toF{I&y&S`vz;wjHCkJ5laQv3Jp1b9>mQFonxzlb@_DjaCpYu+x9AEV-P?P$n0~;Q>*@CV3dN`M<z|cTXVcAWF-O-&+pF~q-JLHlR_{0CPweBJE`M0v%B0-wv(@Dd?xWD=see0NPtopmzYPrG<_43x9MskJ>-zKU#hcmbZ21N?e6)L?YtcjY>?*mMF5b;H$>?^2zfsX<mTcdzq>0UVZ}01?8Rl(!HQnMb{DcuLr`PPq`E;4A?=Io~H9yTTpZkB=ZI|#SnJ?wTsZXRleeiU<S}itD&)MGT+3kFBaj>}>JS`x|X?c-su0e16R^1k)@piV;y}i93Uw-padzKv}&o-O+o24#@w2RfgUC#e<J4;?Yd;Q~!uV1`+_Wh4i_p^V0{{4@q-#`2P^W<!{SiL`xI`V_$1&LUOE=>uAT;49vxAPUq@@%$!KbtL+&x9Yoe)-Mw<hRdXKmSbX{OtMn-y|=;P2z8#f&QO^4KQc(<r{S60_^kI56@r!@%iiI)pvjR{Z}u(|M$<5?R4=LtFn?7Wz`OnKdo+)<!XIBT`WE%%NZ)!Th3+|EOmZ0UA_U&Z70&|{E`Ds&epT(TTWA~glqR`E4A}6C)0zy&%oMy^J}56)y6+<*VFUa+4THv{O$JL{Cu@u?p>}|*U9B{vz^||6Z`F3e0}!n#Uyz>!}M;pgF*-E*#>k>TC?K6`{CvHuMOPOva@&7#qD$prnNHX)Aa?J;_XB~Zf-e5e%zdHSErlpdXlWJr&rU{4Z1a(B=4u|>)RU=4QAxxLvl4++|1US353tAtDdwc4ZZ=%cw5h8JtxUCQpwk=<>ma%B>63BK__n)X!hk9CD>F;{jdcUfY?4yz9Mz@d+}(KELN+x=*D(>21&SFy+_BcC=|I6Dh3nG3K(sYPy)@SsNfQHUEHoo6|ydu^EG(sVs<{?pv7eL{%W@75-0Chw`gz%#+)yG5w~*;<O~xvK0r0!tUv}I5^y;i0x6@>E&4NBtfm)i01|L@F<VTMgM)){A{B!R5BvSQzQqcDj(Rrl@$LPb^qOP=KcBCb;KtRZ4gu92?CG?R#+;A!jEtB}H`cS+^!h1i46{Ofug`Ty$~31xK7aOmVZ6_{azD&SvGYx&(Dm#MRlu6Gj@qQ2gS{TRqMm0L)0^%5-R#wRbvfUHn9-9X{z-;xsUUyF3Qo7PHy;uafU6~#YSpc<ikr18{U#A&;pCj(uGbU<5T2Ku>F;l^Uwsf=w*a$&;3(sU;*!E0G&Us*Kf|0|Eaqp)%@h)Ny4b9+n(Mb4)WjJElWvq-Z#c8(^P4FuSc9GBYY{-3gY8eE2&N#h^XUSL?EH#i-}zG&NaSBog%bK4(<gE^a@OklY`)Z?P7?6Q_3c)_J;aM|X4_LC=)Jw|`hyBs`)PT54ZVXEU*7D^embAsY$-GG%k%Ymwbr7`8!XfJ&}4gir@wvq-Sg8|&tAO#0T>8#F&p*!Gg?!cjly`6<m2()=~pkGz5eF($JZ}jefOMC{B3#o@KZ}XRgA~UC3?Q(a*T>eQcjXE{`=nP*R1HP7e99GX2};{B<E84oTasrqGG$>zWC$w(;vQm`C3|y<0qN#UCb^M)mEpJaQe$;WEKO&>5FU5DEU8YlHY%Y6`il2ZrAf0iUH!k5xoZbn4xjgVpz+!<j-@g?%M+jWl^wrbb}r&Zw{vG_4LDt7SVlUN4I6lCdtM2!_Djn4?si|`8ZZ{I=`@*{yJN)HX}TO>8c)hg0nK8B%|phIU93!W?1PN*=8htveEV{KW5nzG!cO_=p+JjDXTN^KsSEAD<b-bJ$YyzKm=X)_&9qu(N*r#`$+2Sl^u5T*dFVKpndq%<p>WghluIriX#4I23fOG$pL);R;9#Wt!W!T&JteFE+GJ^TSUhs7jscp)Aa{6Q?93*x7Mk}>~ed$xtd>YNBDM&CaZFkjcxJ{K&|LZ_tSWMu(`b+O@Eqijxz0<G-0(-7*KkQMty!b#-pR_@rg|AHK$ZMaA&BKy$7w)^VRKgd$gD>Z2`1t)0}<UsETxw)LEH+oHYK-Ui3sux1g{dpz5ws>FAx5e%En&7}HvMm;4)6S46&i{PQ?ItQ|Uz<Sd;%K7`^vu^Hu;KcAf38ETJ)V!5~dhZg-W*C!_=F`fE2E+H_HV#08f$X1hH^%Lu4kA~&BgUiMA&F0|!8tW&L=m<920@fBwtYHzXm&qqo3u%Wbg8YV>BfV0HYfC!O6aDw}#W(z@V%olZ_wwa`{NWYay-!qH6)*a%bbNE1DpB3+Ea!=r|J_dc?BnIXdSMzPKC`_;Mh8Ety1?kAmVv@K!0cV4yMK>TpU$J2dYjSJ3S)tUSoMSm<o5HQrsrGO^Dmv3;$n*Bgr{hSsga@j3BL!e#`tts`o~E=mNrQ<m^_eB@9XbQPE;?0B0+3?>Z=#W+BtYt)0-Kl>!c@$rQ&YMtf-TB_wCPvZ*Q#*qaj;n;Su>A|C2~goSFnAf}c)K4z8y^skj+ss^M*7H&Z%Vr!xG}dI|G;`ffU3kX$AzE?`H`FSb`bExT0dNka>69cKAj<tA)Jt!^_B(`t1~L+WM$heoV2su0*t$r^s1&$J5MqXbPKs3A2PS+f*Vu`^?c%<1~g1_q9cO&PWHZ7?(Okz=X*#78RZW)vjTY9s0v<+#5pQ3dU&v4PH!dZ_m9U7N$W7QOk;#$AV?bEXUJlMuD7w{`65TKW|2r(>w}|4e${&$`at>6uoSTgT2mp{l|6^OK%r{pB;j(GFDC6{@%HUa=00ji068XV+tK9bR5uN<4_B_x3{qZx$B(mc#__VyjNK_$V`B^v&(f&0-EXTFr;3B%9=QLp}4B_C9ndyo#TaEKTUfI(ZVAgrYLjJ%)Rpu5Xth+6}!7xD7L`5IT_5RkMF3dTP3uU&z3dQ8mfRGO-7QuHn?dr%Vshw92#O_s^cQnjf!ba7CtTM(BP*XT|${3-@#}yLf{Yl9<BK2@+Rex>Qe6ou1QIub+SW-OE>}Up@PQuGil*)RG^fxhzlXwwWYZ%Wp;AwpojnujE_N7CC+^3MpB%d4q3NqtAsbt5t*6fk+;zrh-^(q~0>CvkY>wk$UqwPiYghQg2pPX;GuiB9mr9nYU(lWQEiYAZuBol&Z@DwFP_)Qg>0IFD?6A%Ga{UD_Ej=D%Is}T9%bom#2+B(Bu`W%R?>ghce|z@=Bi1>ZU1CUC6X|MO{{CEaWomylUFI)#^f#>e!mV(S)Mb!G%ylE{0O8!*3O8548=w4mArmI;Tz6<X&AXEnt?L0uxuIsrJ9kYYe9<bt22q<{7_AbtRUoW#J&zl>ws;whXCER*|-85c^9fD-UUo>S|lJCeOL1S(;1diyAYV@q1Bfg)Jt~79%Y)#mZ1gnNp{-!3?GR-s*f{#I@Af*1CE{USjPtJRsjIuD{iZ=@G2SKx`~iX&vaj&_u4QfE8+US@Oc#OsiDBD@hbtnPHZ-&R}%Wf|yyRQc3_><duHcItp}zZB)AOWm<)X3N6*S&FVID12|80g7Y*jxHx&H)0t;^QRA61)73)%LF)2di2+S=eabA;HuADeZR{lmrzBBUMawd|u1VRZP2l%RNv>&ycTjPvV{e;6+N^cW+7>hIbvDWbXpbab=D86@nH8)vSMDv*2-gJt(fO{x!R$k>99m>)Qy3wE`;=C(ik#1~(D?=J1kNX()%i_p%$R&vuFZ1HJiq6`WIz*)6ER{V$2zNXmZ^Py8m*(zI@<=TDc`kD43?6htd-fyvIG}fowagYSz4J!*8~EkN`;|wWuX?7S<3g;X`FKvvO?#I9A|43z}q@MDOQwaTCJmMWvn4r9te;iHRdx^I=@9I3Xutw&TmnLy0LzS%I33z+#~O1+V32T#`$fsR%seSkNs})RNHJIEkvTG+V7^U!G}_3<9q^2$tso!I$v!HuH^U1#!+KB_`S}x-;fLZy-->VSz4h_T4$m381h{B8_nDNW?Dy`R{9JGvElbjX{&D9z&?P>^?4AbjlHQW?R$%^%QJOl^H)-6H;vK-n4fYdfvl5t2Ac=YcZ<I0?^wk`>x3TCcEB*C>9*E>bB!9xr94-`RJ0(8N&)4Z5;Q|el@}(OvaF4|Cofbs6?K}IK}uD|fLH+vz`V3sFN;d_6MCX#+f-GRD`30;md?E5#sm&XzZAt+s5B{Shyp%V6-stl15H;6(3-qy!8!y=B|I)o={<&Ss!YF%Bm;NpGDDo#6d?pu4OO82N&r<vWO7ax098Zy0YXA61|gVugAB-}6r_X6E-GsRr3qbVze)kqP|&N50474)2Gzxi+r|_ZL@G<QX>j3DN{?Hg8?KfBm5Ce3xKsg^=Xp*#)QXE#2_>sS+wd`VSszPls`sKWs(=cu$yNkvlZ|x{z^I~t&<x&Vi2us9ekJupGJ%Q`^T@~q(z2?gp3)c=HB+zyFuDmT1e{k~>RJ`hg{@Rvt3d<4FEXjvs0-vKOisYK9PA{IG-}d7L04(*6+5|Ad0R-S))*5b7oaT6RtRZE;{t$L3so+r6#$##3e07CfgLraSd|pwC{rn4K^mon04~;BiWLHl4FQs<RiH6e#7C>RYR$6ZqD#fCiF?{9nE)p1>$K5&+SY^*tWvIbohr#fN`f7%;@Z@BQBjDo6jse_mXa(Dzyv-PoeB{^>skX7NuV?{sB9`%<pfRUaFJLlDDB{%iV6(vsAk2IXTlG)F>8R?Q3%imlgW_5Cfgu#O^K0fs0h=x(o(r-DFLkFR5}R2WGaj~+O?dtU@d|2+)zkO(o!Udb&={ygpdXrjAVmC<$GOJQld0L3$6`fpoGF(%3W)eS*vOSs;?!ZfyhJcx}4GtKs307Y)i-@x1?c55eSv3`!sjefd$a{fjJy%v3WybA_f*lC4liJ)kH0T@+NHFoIM6$ya_23jND*5r#A6WRjC-$#;_O`2^)Y?ddNlE5>R~_u$ZZ|RGC7B(5MQp*fmiJ34?)xDTB>mPTCQmHhm~Q2!P5I9J~OI3C5KYHmIXC&I4#1Y)b>c)|_Ktf7oSJijXFa0=B%+rP7zmR4I%&H+QN!txeiC;-OXKTT{3h4IBVU{vd3v2S8&gI+vC}kE*kwEoz?%jL+(!mB>=Ht_1p-(Tc27^X@{9|55>h+X5BG1T~yCaZH(eIp+F=ItYW*v>dGBmS!CQqtb?qC4jOtfFQn*TfqTdQS@T##^whTk%HV{bgR9C#|yDdV1=vXFH?~f0Gy5svFHIPZ{jN?pqVR8QCmF_%u=c>+_%Fg0*zWdkVQF14nX??VccX|qA}&w()ZMgTQk_ex5A&``_eQ<3frJI#0-*3^fVFU#v-M3AmS<)o^@Kf^M}5gyn^7b#9dESK+rm&bSI+S%1MQ}AZAp9;0sO(kf8D*2T!@L3WG9J@`Uu3A;WRXl&;txj)q83qXYcfl!{c=(_G9MLCnvFqh*dYQJr#lZ525jNS>MV)d*b+(wNzuW?Fl0jtf*tYGrh#(4(d-W9@nE&axQFL`=<P^iK>i1%dx^K?>DrFf!UwM_i~<gUil2@Y+oS&~_*!OabGqHOCNIN|DNpk~Ea9BDyvuCZyIrRk}7s9^eVb+G}%)pzc!bh4ByM4^37<%rSzAlvx?mlAwf4=iPV;&R5oUeQ!$F?jcCO3UgwimP1<@Vm3=!iTe%29E~>AwX8(pZYw0St45;kyrLV8mU9}#uw<4CcO)S*bma?kB@0OBRLhv-2;-*7;;NSB>Saw^>KL&Ug_$>bktq_))T<g@=PGv#ENzQik;>vyX{2|cCQKkSvs&h~GHHS}%u=o0#YvrsI3Z&06YQeO$|z-Hc(t37Kuw~+>c?tJIheHC9v@XiHAG4KyfmbV!e7gnAcEVC|6=VXjbJq8L7y~nB=)1sLv0QxkXpH^jxHyBeJv9mOt>eJB5QLAX_pJ-)>v{a#~@Eoq;!j>@iI|aG_{s7*@uNFBd-#5r#4TmoM~Jb6-6P4xh{cTQmcq585m2}1Cx<~qR;0YK9(!BMqG7DAqxmFkdY~3A%V88X|)h(D;MUimT~hI8qa+t@|@0B4wr@|WZu#yvd~K<lBcEVV+M|>XF<&S0AB?~3Jd5`v|8;z%@hWc1*_e-9Ew+Hxn*d?s7OpOJnlh37~ml_#Hc83-6#ZV!9VJbT+5Xnn!=g~MU2ZSW-~>M!=W=p@&HjY8bv`Kv38%q!e|B>BbeEaxzW1Zhk#X(X?s4)7^hHVTba`UE2{FaEYQ*eoeW{ITniS}F~Vf13^7irvnEx<xTJ;hF~nqYXrez!)Koc)O^Hgkt?5i!ce;kK{W1#>KqeZZv&>jap@`YtpanMx3Mww8Q?Rf&{BS=~H(f((h|@AuO{0hziwv0*g17}tFVY=ANLh2TAP6kJP&LtLQ@0<8`6mp;D`HZb@s3ub62{LN%|nk3fIqaZjG|YA6~?`gRfR?wGWGkwnMEq$fyVj-qzr}__m(tTW8}dtQivhx7Zp)$wm0;)A~vTm;3#kDh?}VhcF8raV45AODR@Y#C_+=ajVY@N!BCc_S7F3eikLUrG=bhg!5VD41Ri$~q^;aFf<fy3acP?iWx;?h%C&_B02%mUj0?^J%s6|J@=p;9R_2YBF~=V>%vg$sCv*4y;kE<@9*C&+fTR|;orq}<xKXv0F`B`uR$8XC;2k~6I@Vr#2(OkP6%peAIKs4N7*sWnj1lJA5T_zpXCg+C^wx@4dt-VBhG%U7T3ZRYPpxd^KqA_T6&Zt9XvK7IfE23=Ma;uu9HH^rgK1mPUsXt5!F6F%iyjAC7L8FGxT#gdXf%aLl=j@Khm^|lIPXpAdItnjX*r`@7%Q#xt1$|J+$nQe)uu6b3nH;qq&3RM>Q#BH-6#nzaVVu1t?NSk#LX9sxm0TjYbs%5L97hqM=glao<z+v-wDf2GlaHf*boR7uJQ;7B_c>54MrhwM+!yUN`SRxh^-cS6lW&Xwp9|SimQntCZV#5RvZ!IRY+TE4<NxJBL&w&5f6=U$g%dyWh7Js`wpfxdj@J;W~TCuD)e0<sp7VpB!C2|1<9QvVW;?}Ugg_MFgII<I6=|AX#h1q`w8>kScIm*I_HWMah+)m60wzl8j0N0!ZxaW#i&=I<vhB_5Ia3>crms%aK5;eNWZf%e-a`;SEMvM8RnqxAU3Y~AVYG0(n9CtfKm{PbQKIVq>7l#D(Mgl(%Lp1EC`Q|5H%eQB~?jXU^Z{BX%#UJDC;t%F05O)mLByH?|_K;CmDA)t#+X9kw96?lQ2P2-$SJ&sNt@;k6(&ygdyae$IS5wVtyfvaNouN>M?w9bs^<y_nKkjaBCxXZv+yrq5lZpwNN!}(jw><YEY%ay#U`K5?HA@Yjii5knJVq48H5bt;Ote7!R$KqA_JrvaBq`!EZA2D{1RSJ3B>xb!ss?6$w}%+j4~An>u6;8!Jp5(#_2#mlYSty375FylW!Q%o|sUCXi(lv^@A@klB4HCf!SwzWW_p%^UYH^D6B%VFqCqWG7o7oA*kon~*sSIpa`PpLq-*(GYB`CJ-~mz00nNO2gWi;ykzh`F<q>a8{prny}f`2N%e8Cg5Tg3e}!8K%1#+!uB(<*oC+NgnT<Q&AkcRn}K4kB_xt^hs!tJbDvIgRv`7-vYRNla%Pvq7}xPOad$Isi_{>K6=^AQoKWuj{*7#<C}d&trt>;eM#z+%ATBDM_{OZi0s=j?`h5R0FifjZ?Wr8lusTt#nQ;sWyST#$nP}oZcqtq4ge+JbWDiLo+wKIXQ(6n;+bggOtyE!~@^H?44^3b*w!vKDpb6w#ifwN76&AL@Xd182;uctwV30d|HQBIhqPDmdjF;Bx^PSN=Zwnikd%_Kryg{ZYXrg&XnixHdsU!lMx`5KnwKI8c{s(>0(ja%-n^as~K(0}sW~|lR%wUMX*0VtX2ipn*IaQTTR46>)Ug$y+ay<!qHz-uPF$dqfHpnCFdFob$z-A6``&UO1Y>z8&AHbSuV$eJly2~j1Q?n1!Qu|i9fm*dqZxpR-zBX%G?M&-x4~Au`t+uA+N)oHK+PeD6xu+vJA%$t8ocRfHxoI2O-XlyQnF2s22e7m{cZ4jb0ctrj$RbnREh-GM4coF}D6?zAvxkJ~0nzvKT$)<KV}xu|mT{rnAXl(C1EvC*KL{-Ys`Vz&dCv(#(?Lu6UKp%5?J}VzZp4)ac_L6#6$bfq!fMEyt_ims;MCUEgqaS2LTX+1jWP_o(k^FaI0P<_Hj%k^1qqPGwSY#j>=d}3)<kLhV-Ty2C_+NURABDPOPPEJ&t_h#@w%e!CTs~EA=88zLT<FsD^3kEt3@n3gM6>CXsn4~HVYUltGgx|+ZCcgY+Y`APe``I465A~k?bBqan~o+h)M>5V@4N~g5pFmamn_WAlIdTgluOAGnFaiUNmzcYJ*G$faSDRKqdp=yj5A(M0DX`OA8Y)<Y|_E7TVMy3)w+AdMRUg<_lS=+8raTS>y|7?k4#BRYDkmOvRNHoeDL+hXwyn=B{!hB4A~8VBDcIuOLff{3dJZ0I2ekAgMwY^KH#&0u(Hmo2D9~0%X=8tqO@nW^L<XE0C#g7%@Z7(q8CNbB~RC(_nec)C#h5O~_-)g(Sl%l)1&A@|}=&1ZHX%pCjyBDRq{4eWsF9j4K#qDhaNIY-0c_je9VOsg?2#E$-zMT@z)Dyr&{*0W}s#!(!wSGHH=2_Yg8CyDlVSN+93g$eHLCoA7<H0!pvYlC7t8!M~P)RjFq?(S?Xm<619w$OCFc@SF$2rf&=Qh7u<&xU*&b(;aR?zVk-0Q95Kzq^ZyeA=@or*dne4V9(g73hQ4He8;y?BZoVTmMR0QFtb*D=Z?8rUdo89mj{tc{)5N!nnpWQTWB>v0f?3&WK7`u8-R>L@n9y_5q|QhDg2Qlu0`z|ee@i}Zlq^w`&6}Np$cRRiMgn}lr3VLmtv;?@?1T1w~Z48*w&Uie3_MW1J=|zpx~xjt!62q#`8ziC}DKeyu+G)X>6kKv<R5%K_BMErBM`Pp$56NRY+(akf*BUFsXE`l_v@0^u4SJ2!t4)gonubDAm*@VtAu+$TdN#3@^Eo_8<)2MxbE69t^y~6C848b$ZAPMjbm-GECwYNS(H}<63&~wrwkiTBEzF%nY*bhL+nSaf>xrEHTd{g;O98ET`)K5OO1|NlV++ZB1X~jf5uzGJV0Mti0Q2O|uv{woPK+y$&_ZDBT?mmWa(+vCY@uF)OB|=!2p15F98wi%nFvdx42k-?n1vl7+V5?a4$z?U(@zwOZ!}S>FmtWKzhMo3=)rD4Mj^a;r>Gh3GiJnb6(9rt&PF7Tm=qz@|ZUS<wU)=7d@oMujJQ5w6^TrPP&@%oFn5gk0;ZZAZAwHGMH}v&x)e2=G>tlCapd?>cdxASU^gMe5dS6C}eKQ05e$HmN~*ROgwOa$gGSt}&=~^PqK#QrLEEF!BDd$z(|~RmgUYu<V6*#JsM+)oBf~;kPhydZTEv#>HT24|^;QJx=Xm55`rYXuXt0N^`0`hs;hXvL@}CXgyw9H#Q1}OBI6HaLBx_Hi%avkcU*$_@K)Gc*Si$;VSh_o+{Uen=(>pkYs2=nP<vKC`}+wV}g(Ekb7Jyvyo#Hw%-J<$!sCP<rWf!kg_J+>*8jrm-3J~=toPnwgXT>P^7*Vkdo#mFu*A#0hyZsT^wYV38`qPGNx$?RG9H!Wu@#j1FB-kv#n#w8;p^(b~zwgR3Kl-T=VG!G8vGG@lPl-nUYJuULjl8f}*x-!ndC=V?t4YjBf*jJ`P#`LZ&(MfIN>XfU+Bb8sk}vHEm-PU_*;*(^Hg(nK^DUx#4LSfpU)^!_w2#|EzKYwvfmRlRnInuHY~=&U&)JY#RY-t7{>9&PJh))(6wdm#5L?&MV)CXF?i{Ye0n=U=`C3yE6s39m5C=aq?lPmD1AVc>E(cWD^e=l^SFsAG(b4Q#P!P6bsC(FE?S&K!;^&P52H#1M@Cd$P$W!WVt$IN+0F4RBhY2nDn;SL~ZOsv#^pTV9MKOCImw1Q1CE@9BU!u!w0Em(gfUjPi3p&9C{NRZff&KO$Dgw(&FJERtClXH4*Z}_^Qksg=)70FrYY!))UY}Q`eIF33+X4oKT3BunLSgJE275n~Iu{Yth!ueh><_+>BWJ(TuFM1OSvQQh69&Ph&Er&P1%%!4oK|G*b$N687{xN`KkUfCvxnIOQJl61cK7W_TFEN?JZ6^M7iEY%d+6l4iX@w!vS?c8@@2AY$@81C_8P+cJ7hS`o<eRVs0bWaFyv$dKf$6Y@Py_?NXpo}3D|Aa%=J;JD)}sr7_x^BvkxB2BbOJcpJ>j+Tmvn2_6KJ?2MQ23vuJ_grOx3*TmzfQLX9%7F1*8WhvMV4!GyF~N~WGP6Mo?wFdS4UM~#%*^#eSFn+@d}5FpC8=lU7V+qnDr#o%TJWjlsS(Zc^?U*7nbajMy5#Y^l9_9)5B^x~k(Z!HP+%8XE-!5oTIK?I-a2<*46<F53<hfI=U<@Vbi9o4<??Xr0gqf#S^=S><`grdNe=K-;zcTi8I$uGHB8pr68W;$HLKxva1fjkysX9A7}<)L%4DpRtcPK`V^J|7#<9lEjiK$f-;N)HNgFb`VYQ|B)P=X>#~xr4P--e>nbjq8W*sy7&FaQC#uPUKW@4R}HFE@VMhs@fGCNn63705l2Ud8#&*!Z!Y}{20sv2(e)O+3%Nq8Pl56%#?%?0!`&ZVENfI}MR!Vm9I4OP8%{FFtZT2nE1+)F(QB$%^gk;OXAAS!638rI}?l(U97PrG)2x1I=@1uq*scp)B{V3xGOGYduTm~jnNMB`ZHX>)bkqqoN5a|O&W8hwh{(IzI)dfOJA<FSs$#^~FgvS+6KX_(UMXD~xY2RzZ~sc3aodfDj5r)A@qrK2&Mt#-_l(e&J%VxEpxHQkXX;+~Y&=EfII=7DKwxwMH`OpmU;PT#z)n=1D{HkO#i{L`k_PUGomn(nQbr=Ye)>vcAsqE?E+Qp~r4;iPyU+nAzOH62|yo)}CgFSoLu8Vos9*M(u8M3ONyX;|ZqAGk%=4BlC0RuJ`U*G{nXoRY^!3oq+dL@v@-F*^s6(wA|p@!+tS+=BUO5SZ$rw>~_uY)MAJ=^%JKf;;1lW1ei4N~#GlPh<@|Eo1c7TA-nz!tafz_2emVWSM!{%G2E?SO=_4Q$H)otw_ZRKWz<XTz1W2DZ@Lh?JR8?&p-=qvZDw=&*JgaiiP`l;IoXFB|nFz366QfaGPd%#A=sCMR$}BxJLn-MtK03p9ZOFaj_LMrQ9$l#W6o)(AKR#TgJ0?O<VQa34Y{G4@D{Fr|$5S_dR&RaL83|2xfemiy<mlOsdH9PVO-M+D_HLcnS7=<`EtAyeS;3<MvKrea~X_yb}M080;qRV^MVLL)>Hx_sPAiCx$n&Jtml6Cjk4iX&p131Xb5w&i5a{W?66EN;8KU2$eJR#5|~0!}kb|xeSMSZGHE2ShxztXARGKTKZvLbb6b($2{Qj+8i~0VugwInPHV5;KCYcIr_Ao%2%n}VHntx`Kp-eqj=>}@la=8)=v%9sq!u`3(yC~@><6{*vZ2yj+u;rky7>ADLgeiuT4i2^H5oltB!7{Zv9|Urn<9XM9ADX2D}ujJYBnh+-bUYDo@&Gw7Iadp0*7kWprJ-y+YqPXgSFlw^&4Be9FML9qyGwu-*=FKjWLJEkMj;-(_yJQCFQ;U^Yinw{F=pqeJEr#+jMoc|rAL?bO_ZsqFZH?)WHH`+kSYEyZe2-xL}Y%y#h6Nx9#Hm?vPDaL7Wh9Z$nf+d^c!VDYGh9Kck}_^dV^Aeia7T5~4^^ZOVY#zcDU_~B6cecn$$ADU@uoME<uQHYx;nD3QC7R9+}W5P@w?Kh_T#H8-Dw06qWY>B|kv(z!)IZ!`aurm5ob>W!na!R^xeYk<)iSLc@sbHRnl8bAgnDJ!WO3+O(Pw-}*XBM%>(_(#FMe)Y@w~S@mcrZk+7g4PC%@zq|6N~%WjV7K5=DM80*NzVGn8oEqflUmH$sDcG0WTr&UEI7ftnp)vRn~|Mu2b&6njH;CrlW(JJNia55@|5Z)9KY_7tC!S*z6HA8O-DWf7X-It0EpNQQQx46yB#gul)EzBjGB;+)h-xm6&;bCCIwgc>_K8P8dhtwNv@AF?e~o&O5NekC^gouVGCS(`Ov>U<a=j&_2RIvM5%?(22Lhgg{UH%`)Xb$q014Qgv7--^3rQB?HG#gND|y#yxom$tuoUCa2i_Fs@vmfL=f_c_M8P8&@z3!cnW|j(Oihp<^Ugd75)u$9W^}$+?oqq?qr$G+6UYvD}VY1W0D_7X*uTYuc2K8C_#CJ?TL37|PDt)anctuE2QoCI`h__epk~Voul0B3G>P;}eWx_1bCum;(f*(oDhRGmDwGa$rKSChGF6HO#jIDm~~Wm>))<p&aQz&Ivb`Dx4|36;*9%BNcN$rB2jjcLq^bhz~B9C&|=mGZXVfR#0H4e_+`9-8?+G8MJInQba|%K7j{>{kUK&`jlR*<y_=+sz^f(gg+Y6M1-E6Q@rsshB}rt-;Za1GskT2iyMc*Fwaq9P&0STla#XDw;K$1`I?JwAehBNcn^SMo~y*GXd-6*GW1;PZ8x5vT9tm4f#H^++Dg`@Vy-9iy!A1*@p>zb?+E7magAJSKrF;ur!rplB)A_l;AVlBbsv$*U6o_OZSJ(N{)9GOfRf6Z0ppD<ZVN}hQE=O_XT-DdvL4>7)$b(cIZvG4%CRDjTlAD*CU<hMNpIdD-fGW+@Ub_o@7vdnuW##V-SCoAtJ70WLeMN@W<%!jNPOC+-FS$5@{go#3Kr9(^v;Y->`gzKKodZ*%5D2hatagfXw`me8Mh}~jtXw;i)G_}o?`LNhfp}?Dd<>W(G`mM@nP9%QOsl!edwasj^7Q&Tn+D&_Y*#X*uBJ@&v|`WrdX3jT_H&U+{({9J-flDIB&$w2T6HrhL<&&2MM8T2F!!9OtR3vXQAlw5yY==T3%xnqW_=8c7U6Fg@dc~O3c#;u$J03Y&Ek(Ah&hRX|)>tV^K;SPwg4<kh{STLT5a47c3pP4w4|PSnl_gWJQPBtt+|4M+J+An1jex!5TjnUV9p);I^N`RQ=ZKtUU#h2Td}^%vR&+x7;y5M&8t-lZbiTg_c+z+W~j_JyDpiFflCiZE#7iQ!Lt}a$gOxEM72C$&DPuN>BCUwsjHPsaikm4#}Kp--(&cUQ3pxU@?Zpt$M>e7cZ;z%7T^}EgsT|hthZoBlpw{=9iIXFs!Ds?ZKjwn+g^41HB=W+mwjKoC_YmQ>^uzor*V3csq=Q2j_Fyy3oXiJnDVnKTK8XsrRsoaz88)^4xE=@QnY=<Y1stTl9}#*t$W2wYiryzl1T<%rVbbgb(4zEzCx(Qpoz!PCcZaGgR?>NOwTOQ%HF-rS-C{AD)vNb_BB|nJTk9OJGcOwQPW*E**2(*rxqc62W(uD@%*w0SG^)1r5{M9Qi?+iuZ$PYIf@P&hb#7rB|nZ?_6NAM~L~_soy^rDo+wlQ@`sC4`-S-3MJ{eB8?sO<}FyMZ%)=}r8(9ZtHmSaa=w^RT@+P}Nu-&V_5IUOmsz7&>3i>0sd;X^6VMZ0(H~7H?GEMB^YS<ne3~}c{sR04&CKD2k#>Cp7*}qNQOxiB2$oXF)9-%tDdd_g37zcMC*&der#v+2u^cc@sws}$`2-BHm7~qP^r~_DEf-rL7h!G*s5H&x*eg@23eP_bdAw|jxMh*S50}HKido76yXTnMA&e8|j<p{1Ddj#eVxCw=kEiXm<Jo0Q|Bq!&$5(nsi(zgb7qOhvFz-+E@wBC<M8;HW#XX<7k;C4K+4{ozORdh3ddf{B7tbi>#~OG|m}7o4xGnRhYsd45dDUR<WlK-bsN{&cVCGZt+S9_@iTY6Q(InQ`{iGas)g4SGUiamRVjk}*_1ZPT%$^LH#F_<*S#x#j<LNg1X`FZMw01QB{jbK>w~gs_>f;IKvCu}3e+gzWqYBDTZ#Dxh+dYz1!4%xsPL-RF2*<k7BiY0(wV>343B(FZ7Oi+trc|tQy#RZ((D|jR$USbEw*3R$smB>}qhVmSzOCv$#ay012Gx#vR+VPU31&7>rE$Zq9lJ*{P`r7cyq}fc>7`gaj4XB@F;5hS6PN1|c4n7YoPge172Dyl2%mTH>lJglhV$l_AG&8|wytB$P2*8dxqq!-HlO<SjCJI(9pc(gtWU1@;Z?a`OTaXP)-m%Tn46pi#Y&4KQFbc*SHZ8N3A~Im&KrbQs2jg{ESIvxqCO4U4l&QG!=f~X`K?w=@vI{j4+XfNC3x*wPkO3?SZ#X_j5RCo5^JLmpXY{Uoo&%tA3YJbl1pr3J3MM{htesfCWd(iS1OtCf_3*>Nft1%%=Zi$UK*-cY4V@$OS};eI<oc4ROB|YiDAKcps=>f?znhnyxZ_3nIAv4b!?b{Y&kqFa~Hgw#_!CnsH`}4=m#>8$ufx#8Rq&h(7#m7vXP2R4vG_NbEhHL%+9CW6f*tAancVcBtB;H0JKY+uqav550TVznx$YGncd6G1O9T+rC_Zmm&H4-1&_H;sl?h8^Sy{n`CKsHhbSc1TrjtR(oFUt3dvP7KIKMJ-nEmve8jBP*3UfpX{Fq0CRpWmM5}rs@VYz8ZiLX3d|t&o1-td5c7nT~1_4xCpDNF$Z%WBOAlBBApW8;q*pz<m4+^i$9CI5eZKK>F9{sdN4xtI=YXPG?v*(*K+PJCk4@O^SuKSpgN{kMW+1C`zP6vW}3@xifb_nJ%&PsB_iS=)P(=}_ec==mXwN|c-7ohRx5FZar1;6T};!Qp*3o+Z+dESa?^NnL>Uo%TGGpvccue`?_Q7oqiBBE;J1(Uqqu9KNUZ*`Zy01YobIB%cyV@ef_PPtYDs}wvpnf$@)1X5lhYJ1VVE5rEIa65~f%YKVu76YzZ$r%yM^c1fzA-yP;`&GM=W1*PsL$pl9@7l@z7zHee*2?BtJQ5}8l!BQ~D|rIZ+Ro#3i=|vpN-TKpPtLT+uANYO46T-oT){l2M{dDTtnyqN1_G=<ZM^QK65B%di#%^4V=yJQQ~AXva?eugWgEY8K<-2r%yJ)iNh|$JVols{&&6)##jx>03!Zh3?eGYe?RoHM2(O|b=DsNUROOBZKMqzmR<87mjswFBiq*cqMg>}WJG^Bs-o-9Cz#(lyXg$_YwwYZs)A+Tt&^~eng<_3gKiAZb`94Ask3-0%G@iS}a}9PaP44kgOuSt$Q)MyNfVUG61AD|hKiGL3H-n9B+?T2S;Ah4C4Z9kKhl<G4ido(tr>ixr^#cz<F5DyL=iYdsvRpM<OQ_54D5x8^-zvsR%^uBpH9zwbAq)I`YR=0iiSa0u`35|H!??iT_fCKK!;2qJzkd1rTfXW@D&!j=GP%WopK?i(5KQiS;3v7mOFzjuO(~aqAV_dY)sjysjcNHL+uHl`QKXhsD);(p^&aWdZ<Uvj8{Mr^!%FdvwR@yp`IP#lX!5O%Pf7Gqn-wQkD#*P0Qtsp7CtlgouYk-apPJlDWQEN|A-7h^yD}s|&rjTx>z-!{pW=0$T3fYTs;p1>{75Fh*4elV$j)+0CDVQvg#^<1F~~i=EG~D{`4ltJcKq-gm+a*m|MC1!KlERO#Fx>`E=GfwbL_k?t18FCI{x_VyFWZ1)N(v{{`jrFCFQNW6sg~`zg$TO?%C_t&;Hb}rK7{{1zo4zo2&*kO?JC8>DStOqt)rhUuos<v02Y97pt37eQ%4rd8T_=(AQV9>5aU6Wq-bWH(k&9c13%`OZP5ddyUZIgTImGa{g|%&zBO(J4NH$g5J(P@O@pk*ZK~@(daz+^2;P2KQ6}I>jb3(Qk*Y_naOLM1~WYSpf9UouVxqcJ&jW)Z;6R7Dl2<)!B@)gB|*06+VHSZzSZnJ`Cmzq934qb-5bs1l}31)?=+H{x}W%v{QBilk`LdAXB~WUlw<?^YR>nB@vTX$>UcjpnfS;2WKdFP02OrfaeOs@Ji40iXXB?&^CyedN%A-uWe4emX4`~0#1p<8W_&EY+O4qP+v4!fo96Couc)4ARfWAM>oU&#V8r)6+urj``tfJVbnq@BDY8qef4QA5oLGCW5c}ioe7joD|B80r&DQb`SbG=XhVSjM7r1iK{gpxcLMi@Yp-8B*8WZU=iAI_cim_LRkxHDR@4iAXv*iR?-+52q9hz1aK1L0j*>rt=wJVjB<Jr#m<luaD^TFN%$G#am$M^n`JatMQe(m4GEW+0jCPzMK{cwi^Sm@56;+w#P9XS-f=L`SJ3wTF-W};IHvYqTc8QH<AI6j;ROZ2|>s90X&gu)%ub#JVka{09f+Oobd@hG2=L3YtXS!Ne5drUO$@%hD1TJtsfe~k%%-rZrv@m;!(l}on=g8CRWo;=n*iz2({@z~Yz;{ivWBv0aR<?y$<|63UT)^yVt$p@`S-}Z(L`|p^%VI8g1-~F5pe*OsaE@$sg`8q80jW3Em9KJRBdUm~9e>jwvbdHn#FMA3cO!;iFfb5;Fmvg@2&Xmk%JKgdvchmDLzGaMW_}b2|`7YS?Oy0Ac@a0?|^c`Hz+1k#*IcqacGU)c^g6~G9!rGkPET$i37e^^yqQ?#%om)TG%Qr{%%K6v&zPt3N@J$-V_r11r$<iPJdjPXAy<Kcif%;R)9it_gK{xB0|L&=gW8NGcAA?*@9tSlO`rLeS-$S1#U#}Lo*UQZz<lbz_*RNw-e0e<Qa*pOeg`7A2VY(0@xwtsEuLD=B_ZO@8%TvC)89hNqgm}TPpk%21qldbFzPi4--Of(!ZP-Whq`#{hU%MCU@qN?_hz(g6NMU(zxCttm0Lmfx@%WK^e`N1xTy3WdzA2CLX}x-XJXd23A2~--R1J-J|81rx_HNl{$>r>Qa<hT}y%@gIjju$9006npFF!~<gr8ndm-Fd4-yXMKeE5y^(<&1N+$7uS+gY-bcf>8v^^L_O#{^@1hYIoRa&`OW3Y5y~&u2R2?C_EHSKb<HGhi<RT=3Pm7^RKy2_2fpGIk&EJ;R&6c|*xKlBfB?&?j2aan6tV0Z8#F+4|Z_9F`|z{E$VtM1trlDD|mQp6XSkAB*mz6l}Z_lHaMklyBIT{GjR8)azX~cwC5e=Xq&1JIH@wPd-n+<x9f#y_OgA&H0+5gEe!eW*adcz?Sc4)3<STlIa;=iYYG><eDtsY_D|1HnYV{$X`_CPb2y&5Nh+Km}UvS*ZDqc9zUgvZF+?~bc#8j=%;0$t_OVfD4J1L5PGlbG`A9~DBt@UA9*`Z_380GxAxeRydL+BT!=Uw=aD>}9yH#wr&`DINft1J-821hkFx!)it*#_rzgFtAA4U1@BGBXsK}XgBr@qO_oQpcrG(Wp!oRLPOhioMLfBj?(9NM0!0L*?<?As8Ier9EnEmwVgzqy(y$4+AQ^ER$Xkqorto;3}y;li)5B_E~9_~~(Nwk*3ohpEDX0s*m^ga>#NOT>3)fdA)TJYV#s9^`0%6qO2{Cj8L?Mw&yryo_U%tWppQ=FTBB~r&<K_6cmt^J=4-zd&uXtSqR^Ubzf44?7?tL@*8tlk}0$=mXRgJ19Vtyi@_dX)SipAO7&zU$qbU%u&4g=+LiCwGcjYolY7KiFJD9iE;oFzg8{I5?ZX(XS_Loa3HvPS0+)+tu>H7k7(FKOQ`-mGx>r>6q&M6|Lb^=MA-An^BPUEnVvCBkAjLwvQ)Hk}vfeKli6z>G#W{NWVg1Ct@$r=&&XZcjt9#3n~GKXY%pj^|Kd0JU{*Iv+us0Bm?Nx%OA)0Ot_dG=w{9et<y4;bZ<C6{r0euR`<6rzx(DRwU&Fkll+mw#cmgdKKYJ4A!^rYaeQ*D)tn4{`8eG@x;+&|+>=9bzBPJq52Fku;XuaS%+}{LI{5ByF{Rztz3Xh*&;7bayFR5oHBqyi{w-wk(Ia}UkD%-a^YMw#uIF#A@USS~{2+bIAzD|WJ$JF5zCS%%EpLaf{_foN-@`B8Y?E`;58qT?SDe5%xaHe~UtUkIrpe{vc5{`mB7HS=vVFhuh=SA^?R#nGexG>k@}H1)fwUHSK_Z<vspAtIW{GO>h1U!RjNN7Kf=XQ*_Xma`V`4O6e8=Nfi}t4P&gyfMK|BK9=iu$s14$;6L3hp`?R)OxJC7MXg1u0S8JT7@ByYw8GSb=b_l?8#U5NSHSG;cf+uVqjN;-j~S}Uqw{kiY*P!tJz{7B}E+@ji5zTR0NDJ$SoU;p0mmd?3b^xNcm1<GO@UC$G+m1&LNB$u~qc~P_28XGilyP0{!h#?`X2?Cx*4Cs>;;P?3&rXh)clPEy{cJZMz#gxf6Y94y@WF%V|@|WDe!goQ9Mb@zI12g|4$_mbdn5dn(2Rm&vv@o7z<EQ*FQ$+(seJY-<tu5T+H?wV5L7G}Yc+j)cv+44!`)QAvQW{yO>(9WMdqnM{bw2jq9o`c=`f0?iS^56-Y5q%Ug)JKHuaTQj!y2)KB#F&Tr?oS$#>sz1v_~lMmGgtR{&~m9%KAk$)(|o27ATuJxj)dR#KSP3e$wVgK2F939?@g=hHV5Si2Y@+XR90Xzt+L`(H{n-fXmy(LOVOVzS({ljqLI8P~Z>c<$0}~$|%mkpn(2<crs`;9BQi)B=1kraCRS<i<{|uy^)CLvlp*VUwre!F-FB|PhY)Yl+VBX?nGPB*zSDBXa1HR4z=K&2rwQWkJZM#o?d`-&R1)F^zWSn7rvTMM!;bcdj@<_V6H~_q~t%v1TNV?uv^(Lj(kc6g7roQ#F1Hp9Z4M@>Z2z^Q5^eqvpJR-IeAcO_r`IoU6<t$U{Em5RIeBVl=GzoD8>03%%0Y<tLlH|PhWpywkOFwLyes5@_3+DuQ+ZBIS7{R`IS!@DQB`nO-i2Q@p0Olzg3*P{?rtmmHUz9=xTwH`DVV$<kLPgiyk@~AYzP6&U7~q^!r1vTgzMB9N18ka(Gx;Z$wwu=4EJ^%4`g00{uB#ucjC0^xn2Bo0*QOL84^P`(|WVk*v?KsP$5}{KUJMFU7rKzrCXI0a^y@vL|4#DK|dFA_$LJj>^~y4ix;~OkE##GyxwJogl)lJYr+`!hd!g+DFF0_%+fo9^eWC>Mb-tjsYP5`TM}MHp_zB0nsEF*a&8}Y@yzlvx`#^<fo>%hZc;KHyPlXT_^>{gIbYhp~#S&<U7ywqfWDU%Xdxx|0^!?|4R?~U+EzKSNX?1A6NuHFW|~mvsHI$JsD2gZsf}LC*~XZ(2tHi6xJR}bf@HAgLfmq|5_*A!l~BjT`$=Nb(d0Zx2wy`fd#4|Lph_n&q~MRy}R2N>y@la_H>jzq51OD&1ir0mwl~n{5Z?*d1^6#3%Po=T3u+AF-|LGRQ&RI#4WogFch1|+RpfHkU@ubcM{&tX69*~FXlHR^oVQvgxAwZ#8&cV<cM2;pDs=Ysnk2d)QF>Aw_+TYH8SQJetv|z6>^}v*kktj6Cxmky0c_|r)$h8pV%soAOGd?Uj`MpBz%0-tLIY#AfG3%AiwDjfw7X=&1Sv;rCv|rxlkycEp9I_H_CI@FjFqktGF?rpwumF+s=pbt~^zm6E!t5(6lX<XSSc^1NSwK;4#;(;7i6+zT8(XVl%Hd?CJ@(GL^Z8ibAtrJl6b=^M8L&LqzwqQx%`Y%EJGySQfK4(}neSV;&@3UybCm#CCP6TYLO+{o!=8ou0on6Ma3Gpg*5*uaTv0R@*1UqIpLUE2Pg#E(ZhS+R)LU;yXUOb^q(%KL7eZPQQNk-FK|dGKpY6Nf(|-MwkkoUI;J-4%p6@YHPTf=-Pq1|86zEfWPrU_keF^5JDI5$t7Ym)vX+{04CzkW5v0XM&Et*)A?)x9l!0>Hov}}UCb#Qb$V7%q*O<${pNJ~=bPjT)U|KJ&{&maNp+j2=eL_HiEEP9>cH)DCa}5g8E8n0jt<g`toAQ#j;y-J^6DqPh%*gxO+uM9L*^e&e4I$%NK(2Z_xUO$A`DtyaN6+AYP?HAh>UxOUogMfx_403!&2jWTh=C+-}JM@u-8+@`q<$oKO;MnUrPG*orkOoY24$DIcpMYSZv6gQfN(foh4-Z4L8T9H=rEO&H|>>d<%-1qZx37i6fsEX*fL8Fvi}ebICCu(D&Wp4CvwT9zyu?$gb=t<R~4lzG>T^S%95RgjPE-oSt2J<K;yfn*-y05^af&9dtAFx><f?mi9in(!1gr{drJ6i~Tflx$hReIz0muzNb~VIf|c$I)$-P_uzC$J0+6;t@S?b)iVC)Mekk7ySZ8SYV!*ASzt1h$>Rs>_$M?~FE**8TdXv~*8R|PE?&R*<MY>43!knm_U?$uv=V`@4GF2Lt&&;_bRMWHGFW;y>v3Or45g{>acM<|cSQ1&BaM3RMJsOh4js_temok?qD;3reX_a_a<B}sCs;p0a(6vd3KfsH=W7g`jQl1^E0}f7<{|%3yu~CP-K_bItU7h#)pV0^R|MMYd_B8{zMC$R=YLdzb~XhOaLXmy^b?<waU7K_g-wsmPMi{*JFjN%qyEgDaQCHcu>V7@jP2>;S&QA=v51D!3y$q}zPz1v6)fM)1{~{@sl=~|2;0@M)^@UQMb7&E8%dq>)vF~^|3s%W3III#MBCi=a;$6IGe3T~UaerZz<`E9yot%Ho2&Kh@@=wRy=R>5dN$qMt|d+d<#eeTE$1IP+ha5TD;L!J_Mmi-CFoW@3(=hmDm27W&eF#t*0@iVH%N{&Eliz}?&u;`%O7`gMj*y*w)##NJub;iDK)Ju-|@IU(bL*-v*IM)?_50wczLgjHR$S~6TrGCk$rmX?F_ps4v(vj;j?`m<4IN;xUg>W*{7X$>4iXM%TulEfvP*@Bx_E;Xwvnk*Cq&0IoBF@zNVd;?Hh<W8v2D@`<<9nihkn6Hr`jDYifIi8M?AWG9DM&74ke1_{1cAF@bD<OkL27xi~!k!3+C&-AKzbxP05{CsAP6?EaA|kDW~0qrrT_aOs|)#_EgY9*2HK|20`H-NU$9Om8<5VV0iwDG8d8^X(S!P%L*$DvmIxq;GktN^pV@+q@KOoDBFydeFoftM0}V<k={>Hm7t|PEAvP0pc3>lD^k(=F7<Wg{xPmg0Z+Och@P+AMdU%%IMyI;?Dh8BkD?EU%HZbqHS_%d3{u#S3E=6gtpvSH!DAJ?WYw@Ox6$fs8PU;#7nH=8D#D~nPH!O_Sp~DVu+osZkHD`X7ODt=4buQVT|JY+v`^!xIulpj5~DQ(ST(?L@FD3|MaHgC+;=sjyKb3N>5|xDh1V;=UkP2lNYnxoha%WSoYMXbdw>#<(mWD^&53NlUceyY^WzWZ0~)0%SXBrPZW@9kHcXva(k`z#fpxyhWB?SYqGOxBRlPzokz6d!*H@&T87MY2WjU^dpho9XzND~4=wu<ALg$4X!#@`1U!B|d~Q^DU)acOXbT#B%sII08}!BHNBx6)ZO-^KE#>kdce)>6>3gXves*oHpRuimRmZ13#kV`ZT0y+@X6`U~*`;9Lb3Q;TihTVC5dO{W*<ya4*yhSaDScn{;ml`74+@|i*rSPz!*bv$SGznSIY1yB&5Vcxh9^&DKW^s71~6g1L*iU&o$U7B{K!M^sKaP%G%+3~s}2%c`z3muKA3jN7|Mo8r{X@{De3;bc;}x+(UC27j|c9^wtEwS?+JAtm%}}mn>+bGT?Jpt+YWS@mLEoUl-urx&Cl$C-0>UkRQQn{x}Ct5Z+!0NLD@hp3+mnoIv%>a`)JSU-s9fz%*HcuQ`W16MijTRLkU%i9{_H;UP1e6dXpY6@hIeS1#14yPGwzxHwkp*{F-siKi}eE)_Hy<r{K<~qQuTV2ncPvl|U+g8y*F*@^`}_gZRqRfswmE`0W6m%W|}{*JA?^r@RS|wUxW{{fS~$s9g!IJ4-aIXkaSzd2YCq@(Wk}A6W6;*zO5!^`Dt|T+96bf4`*;p1qsFMp)i%DAZIaiCZYjWb^cVdb8;r*9E;_&gQbXq@shpU+P20?vCz_>fZz2Q{CPJ+ramUukm2lKc>CMLr2Gh#CJ<?8W@Uhu*CyMk9x_x59gmfxmW)0n|9dE+(lce%_u;aiZ@$cjG%>Y7u&<+w*X$t=bt<6_8m{Usbg`uUR~Qp1N6v*ft$sxcm(JO53DRa(J+}pe=N891n1B-**~tQl0HlL9QDt~_2=+kbLv7TtCz>v-{1L>Qb_}Q=hLNnSv;$1+cMk^5>Je0bZ$3t`Y(B7Qf!)Vl@4|{{;j?D81^X)vFM>uulR5C<q{P^Sw6a+Jv#ho)_8g%$Bg<(;z}NSc1&l|@y;hxbI%sNd%M9uGZwdm*q?v<NYed%x$*q~-kmXv=zX9&ch4F8vk448+r8rC?iu`tZ6S?cdp!1TJh^AH@0@0Ts6)kxoMYSV!sp2kzC`hW8ET!MF?nLckc4C=_s*lKjkMW0d$y~l?n}G=xb1a+Mz$>=!#TA0?#mpRtJC|p!D)$)*;6|SHy_DE{6{Rizu#-x*$U5XyXoTa-d4LjsV?b9lF?2MAMB+ii87cqDJJD4j6D<c!AHb{)ojJ}b`B+{F73`-Pkiz}mQp%g`{l*_Lg?mb*!!RDQI{YZ4Z6{&SKk-fXT8ec7v_$t-`(%r(V({RQ%$V%-^UM+@S%f|a$5ALH;P|gLw1iT&^sh2nS6VDu*h$?CrsOTb$KaUC5$`m$LKPabb395V&16op8H7wIbhFZgamdvtgwSURb~vM_xZbX_@@hY?gMia&<^Z;J8G;|LV3>Li1I&MZ4WH$wO;*IPEP)s;rMHY<F6Txzh*f8n&J5WS%zbm%Q(;ozveRjn#=g#JeTobk++yVkf~Vwnw$7vo|{<ycg;qu{)r64p-}xb1M$}k#Q(7wh`(kV{+ez0YqsI9*@plBWE=jHWW!%`3jfFE6#kR>gJ$D>Vhj~yhe!2BvSjXyzOwTlvt$^#RtsGx;QmwU{cd`T_w{M_isYYl_wVWc@Y#R3|HEp2egK}e;UoCPw)K5O8CLR-k7ZcTC$2`<LGBwI@lW#n`)`t$-zI;0`G?oZ*Dt?$t}q7ue*n+#lp+')), _CHAMP.__dict__)
_jish_new_half = new_half
_jish_choose_action = choose_action
_jish_on_hand_end = on_hand_end

def new_half(config, memory):
    _jish_new_half(config, memory)
    memory['guard_uses'] = 0
    memory['champion'] = {}
    _CHAMP.new_half(config, memory['champion'])

def choose_action(obs, memory):
    started = time.perf_counter()
    cm = memory['champion']
    before = memory['compute_seconds']
    decisions = memory['decisions']
    active = _CHAMP._observe(obs, cm)
    _observe(obs, memory)
    if _CHAMP._profile(cm, obs.street)[0] > 0.65:
        memory['guard_uses'] += 1
        action = _CHAMP.choose_action(obs, cm)
    else:
        action = _jish_choose_action(obs, memory)
    memory['last_obs'], memory['last_action'] = obs, action
    active['last_obs'], active['last_action'] = obs, action
    # Charge shared computation once, whichever policy supplied the action.
    spent = before + time.perf_counter() - started
    memory['compute_seconds'] = cm['compute_seconds'] = spent
    memory['decisions'] = cm['decisions'] = decisions + 1
    return action

def on_hand_end(result, memory):
    _jish_on_hand_end(result, memory)
    _CHAMP.on_hand_end(result, memory['champion'])

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
