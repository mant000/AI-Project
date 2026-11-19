"""Persistent CPU worker pool is initialized before CUDA; each task remains an AST-only rule."""
import resource
import signal
import time
from state_feedback import intervene

BANK=None;CFG=None


def initialize(bank,cfg):
    global BANK,CFG
    BANK,CFG=bank,cfg
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    def timeout(signum,frame):raise TimeoutError('Pure arithmetic feedback task exceeded20s')
    signal.signal(signal.SIGALRM,timeout)


def ready(value):return value


def measure(task):
    index,expression,mode=task;signal.setitimer(signal.ITIMER_REAL,20)
    try:
        r=intervene(BANK[index],expression,mode,CFG);r.update(bank_id=BANK[index]['bank_id'],instance_id=BANK[index]['instance_id']);return r
    finally:signal.setitimer(signal.ITIMER_REAL,0)
