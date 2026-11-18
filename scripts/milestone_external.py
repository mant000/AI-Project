"""Relocate pinned author reference assets without editing historical metadata.

The original solver wrapper is unchanged. Only its metadata root is redirected
to a newly written, root-relative provenance directory for the current machine.
"""
from pathlib import Path
from stage3_common import CODE,read


def configure(root):
    import external_reference as author
    directory=Path(root)/'operations/milestone-external-v1'
    assert read(directory/'results/baselines/prepare-v1/completed.json')['milestone_portable_setup']
    author.CODE=directory
    return author


def lkh_solve(root,*args,**kwargs):
    return configure(root).lkh_solve(root,*args,**kwargs)


def LEHDReference(root,device='cuda'):
    return configure(root).LEHDReference(root,device)
