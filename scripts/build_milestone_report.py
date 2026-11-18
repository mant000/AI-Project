"""Build the course PDF and editable DOCX from verified results and one text source.

Requires genuine Times New Roman TTFs, supplied by the user's OS. Fonts are
embedded as PDF subsets, never redistributed as stand-alone font files.
Every body page is explicitly bounded; a layout overflow fails the build.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph,Table,TableStyle
from docx import Document
from docx.shared import Cm,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import fitz

CODE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def p(text):return dict(kind='p',text=text)
def h(text):return dict(kind='h',text=text)
def table(rows,widths):return dict(kind='table',rows=rows,widths=widths)
def fig(name,caption):return dict(kind='figure',name=name,caption=caption)
def fmt(x):return f'{x:.3f}'


def content():
    folder=CODE/'results/milestone-fragment-expression-v1'
    s=read(folder/'analysis/study_summary.json');m=read(folder/'analysis/mechanism_summary.json')
    strict=read(CODE/'results/milestone-fragment-v1/analysis/study_summary.json')
    assert s['status']==m['status']==strict['status']=='passed'
    assert read(folder/'references/completed.json')['status']=='completed'
    proof=read(CODE/'results/milestone-reproduction-v1/release_verification.json');assert proof['status']=='passed'
    effects=s['contrasts'];costs=s['costs'];strictcosts=strict['costs']
    lineages=s['lineages'];valid=sum(r['valid'] for r in lineages);validstrict=sum(r['valid'] for r in strict['lineages'])
    armstats={a:{k:sum(r[k] for r in lineages if r['arm']==a) for k in ['valid','unique_programs','unique_endpoint_route_vectors','directional_feedback_prompts','feedback_states_nonzero','feedback_states_total']} for a in ('A','P','F')}
    ref=[]
    for label in ['soft_penalty','centered','phase_centered','LEHD_RRC0','LEHD_RRC10','LKH']:
        x=read(folder/'references'/f'{label}.json');ref.append([{'soft_penalty':'Manual: soft penalty','centered':'Manual: centered','phase_centered':'Manual: phase centered','LEHD_RRC0':'LEHD greedy','LEHD_RRC10':'LEHD RRC10','LKH':'LKH3.0.6 (1000 trials)'}[label],fmt(x['mean_gap_percent'])])
    prep=CODE/'results/milestone-fragment-v1/prepare'
    pc=[read(prep/f'{phase}_completed.json') for phase in ('pilot','full')]
    mechanism_wall=sum(x['wall_seconds'] for x in pc); mechanism_cpu=sum(x['parent_cpu_seconds']+x['waited_child_cpu_seconds'] for x in pc)
    strictwall=sum(x['wall_seconds'] for x in strictcosts['attempts']);exprwall=sum(x['wall_seconds'] for x in costs['attempts'])
    discoverycpu=sum(x['parent_cpu_seconds']+x['waited_child_cpu_seconds'] for c in [strictcosts,costs] for x in c['attempts'])
    peak=max(x['cuda_peak_allocated_bytes'] for c in [strictcosts,costs] for x in c['attempts'])/1024**3
    f=effects['F-A'];fp=effects['F-P'];success=f['predeclared_success']
    outcome=('met' if success else 'did not meet')
    pages=[[
      h('Learning TSP Heuristic Programs from Search-State Values'),
      p('Milestone: from edge actions to rule fragments'),
      p('Hongye XU (72655146), Group G042 | CS5491 Artificial Intelligence | 19 November 2026'),
      h('1. Question and contribution'),
      p('Can search-state feedback help a frozen code language model discover better TSP heuristics? Endpoint route length provides a trustworthy selection signal, but gives little detail about a rule\'s decisions. Our original proposal learned values of individual penalty actions and supplied parent/child value differences to the generator. The milestone tests a more basic prerequisite: whether the measured intervention describes the object being modified, namely a repeatedly applied rule.'),
      p('We contribute a matched-work rule-fragment intervention, an audited bank spanning 128 original instances, three trajectories and four search phases, and a controlled downstream discovery comparison. We separate feedback informativeness from useful program discovery. The work follows the minimum frozen-generator core requested in the proposal assessment; a new learned predictor or generator adaptation is not required to establish this comparison.'),
      h('2. Search engine and program interface'),
      p('We use Euclidean TSP50/100 data from the author training release [3]. Guided local search (GLS) [1] alternates best-improvement 2-opt with one edge-penalty increment. The fixed work cap is 200 sweep equivalents: 235,000 and 970,000 evaluated 2-opt deltas for the two sizes. Best-tour quality always uses original continuous distances. Gap is 100(L/Lref − 1); supplied references are not certified continuous optima, so negative gaps are retained.'),
      p('The model generates a single arithmetic priority h(d,p,mean_d,max_d,lam,progress,stagnation,best_ratio) for current-tour edges. The maximum-priority edge receives the penalty; conventional GLS uses d/(1+p). The solver, initialization, lambda rule and numerical language stay fixed. The 96-node AST permits only bounded numeric constants, arithmetic and whitelisted numerical functions. Features use the current state only, with no future branch labels or reference-tour information.'),
      p(f'A strict full-function precursor had frequent bare-return responses ({validstrict}/384 valid). Before inspecting either new dev result, we documented one uniform transport revision: prefill the exact function header and return prefix, then sample one expression line (128-token cap). Both complete batches and costs remain public; they are neither pooled nor chosen by utility. No further rescue or outcome-based expansion was performed.'),
      p('This is executable heuristic search, related to EoH [2], rather than a replication of its thought/code evolution. Deployment runs the selected arithmetic program without an LLM or neural critic. Only actual discovery and program-selection route quality can promote a program.'),
    ],[
      h('3. Matched interventions and data separation'),
      p('At a saved penalty boundary s with remaining work R, all branches share state and RNG seed 71. Single applies h once, then standard GLS. Fragment applies h for floor(0.25R) deltas, then standard GLS. Persistent applies h throughout R. All end at the same total work cap and retain the best original-distance tour.'),
      p('Define V_m(s,h)=100[L_std(s)−L_m(s,h)]/L_best(s). Feedback subtracts parent value from child value at the same state. Persistent outcomes are a diagnostic target, never model input; they describe continuation from s, not full-run generalization. The 25% fraction and 12 diagnostic rules were fixed before outcomes.'),
      table([['Role','Original instances','Use'],['Bank','128 (64/size)','State feedback and diagnostics'],['Discovery','32 (16/size)','Candidate endpoint selection'],['Program select','32 (16/size)','Choose among four elites'],['Independent dev','64 (32/size)','Evaluate frozen lineages']], [.23,.25,.52]),
      p('The 256 previously unused originals (nominal value_train rows 577–704) were reserved and audited before outcomes; IDs and coordinate hashes are disjoint across roles. Each bank original follows standard, centered and soft-penalty trajectories, sampled after 10/30/50/70% work: 1,536 states. Feedback uses one predetermined state per original, balanced across policy/phase; all 128 contribute summaries, with two fixed-schedule examples.'),
      h('4. Mechanism result: more conditional information'),
      fig('mechanism','Figure 1. Ten noncontrol rules × 1,536 states. Positive rescaling and standard-rule controls are excluded. Correlations are pooled descriptive statistics against the persistent conditional target.'),
      p('Nonzero outcomes rise from 82/15,360 (0.53%) to 1,538/15,360 (10.01%): a paired original-instance increase of 9.48 points (95% interval 7.79–11.26). Tie-aware sign agreement rises from 0.504 to 0.573 (increase 0.071, interval 0.056–0.087; 102 informative originals). All 56,832 branch tours reprice exactly; 3,072 standard/rescaling controls and 12 independent replays match. This establishes intervention fidelity, not improved discovery.'),
    ],[
      h('5. Does the feedback improve program discovery?'),
      p('Frozen NF4 Qwen2.5-Coder-7B-Instruct [4], without adapters, receives endpoints alone (A), plus single-penalty feedback (P), or plus fragment feedback (F). Matched settings: seeds 1201–1204, standard parent, temperature 0.8/top-p 0.95/top-k 50, 32 outputs per lineage (384 total). Failures/duplicates consume budget; feedback compares the previous candidate with its actual parent.'),
      p('Top-four unique discovery elites undergo program_select with three solver seeds; all 12 programs freeze before dev with five seeds. Seeds average within originals. Paired intervals jointly resample discovery repetitions and size-stratified originals. Success requires F−A≤−0.05 gap points and an upper 95% limit below zero.'),
      table([['Method / fixed reference','Mean dev gap (%)'],['A: endpoints',fmt(s['arm_mean_gap_percent']['A'])],['P: one penalty',fmt(s['arm_mean_gap_percent']['P'])],['F: rule fragment',fmt(s['arm_mean_gap_percent']['F'])],['Standard GLS',fmt(s['baseline_mean_gap_percent']['gls'])],['Nearest neighbor',fmt(s['baseline_mean_gap_percent']['nn'])],['2-opt',fmt(s['baseline_mean_gap_percent']['two_opt'])],*ref],[.72,.28]),
      p('Table 1. Same 64 dev originals/five solver seeds; all four A/P/F discovery repetitions. All manual rules are reported. LKH/LEHD [5] use their own budgets. LKH rounds input at scale 10^6; tours are repriced continuously.'),
      fig('discovery_effects','Figure 2. Each colored dot is one paired discovery repetition; diamonds and intervals show the joint paired mean and 95% bootstrap interval.'),
      p(f'F−A={f["mean_gap_pp"]:+.4f} gap points [{f["ci95_low"]:+.4f},{f["ci95_high"]:+.4f}]; F−P={fp["mean_gap_pp"]:+.4f} [{fp["ci95_low"]:+.4f},{fp["ci95_high"]:+.4f}]. This {outcome} the success gate. Four discovery repetitions limit precision; 320 solves do not create more discovery runs.'),
    ],[
      h('6. Interpretation, engineering and cost'),
      p(f'The expression-continuation study has {valid}/384 valid outputs. A/P/F produce sums of {armstats["A"]["unique_programs"]}/{armstats["P"]["unique_programs"]}/{armstats["F"]["unique_programs"]} unique programs and {armstats["A"]["unique_endpoint_route_vectors"]}/{armstats["P"]["unique_endpoint_route_vectors"]}/{armstats["F"]["unique_endpoint_route_vectors"]} distinct endpoint route vectors within their four lineages. These are within-lineage sums, not global unique counts. P/F receive a nonzero state comparison in {armstats["P"]["directional_feedback_prompts"]}/{armstats["F"]["directional_feedback_prompts"]} of 128 prompts. A richer diagnostic rule family therefore need not imply rich feedback along the generated search trajectories.'),
      p('Three F lineages retain standard GLS; one selects 0.4d/(1+p)+0.6d·mean_d/max_d, an interpretable mixture that softens penalty suppression. Its dev improvement does not establish an independently repeated discovery. One F−A advantage instead avoids a dev regression of an A-selected program. The next experiment must control the common proposal pool as well as feedback.'),
      p(f'On one rented RTX3090 24GB with 14 vCPUs/60GB RAM, the nested 192-state pilot took 53 s and the additional 1,344 states took 216 s. Mechanism preparation/measurement totaled {mechanism_wall:.0f} wall s and {mechanism_cpu:.0f} parent+waited-child CPU s, without counting the pilot twice. Strict/expression discovery took {strictwall:.0f}/{exprwall:.0f} wall s, {discoverycpu:.0f} combined CPU s; synchronized generation occupied {strictcosts["generation_seconds"]:.0f}/{costs["generation_seconds"]:.0f} s. Peak Torch allocation was {peak:.2f} GiB. Input/output tokens were {strictcosts["input_tokens"]:,}/{strictcosts["output_tokens"]:,} and {costs["input_tokens"]:,}/{costs["output_tokens"]:,}. Labels, CPU feedback, references, failures and reproduction costs have a separate ledger; matching output budgets does not equal matching compute.'),
      p('Earlier Transformer/MLP trials, grammar-derived SFT and three-seed terminal/state GRPO are preserved with all 28 inference exports. Historical SFT showed a dev improvement under its own protocol; it does not establish state-feedback effectiveness. The present result is directly measured feedback, not a Transformer success. A bank fixed to three trajectories and 12 diagnostic rules also cannot establish arbitrary-policy or out-of-distribution validity.'),
      h('7. Reproduction and route to final'),
      p('The public release includes the 256 inputs, snapshots, all prompts/replies, failures, program hashes/lineages and raw routes. CPU-only commands independently reprice endpoints/branches, recompute selection and feedback, reconstruct all bank states and replay selected dev solvers. Verification uses an anonymous fresh clone and a separate Python environment on the same server; it is not a new physical machine. Full GPU regeneration has a pinned-model entry and a new output root. Report facts map to exact commands in docs/MILESTONE_REPRODUCTION.md.'),
      p('Next, isolate information use from changing proposal trajectories: freeze a common, behavior-diverse candidate pool, compare fragment-based screening with endpoint-only screening under measured CPU budgets, and evaluate selection regret on new originals. A policy-conditioned critic is considered only if this oracle screening is useful and predictive holdouts pass; old edge-value weights do not represent this target. The dated roadmap specifies pilot gates, resource caps and a freeze before independent final testing. Final data remain unevaluated at milestone.'),
      p('Code and grading release: https://github.com/mant000/AI-Project ; tag milestone-2026-11-19. The exact commit and verification logs are recorded in SUBMISSION.md and the release metadata. No Canvas submission is claimed.'),
    ],[
      h('References'),
      p('[1] C. Voudouris and E. Tsang. Guided local search and its application to the traveling salesman problem. European Journal of Operational Research, 113(2):469–499, 1999. https://doi.org/10.1016/S0377-2217(98)00099-X'),
      p('[2] F. Liu, X. Tong, M. Yuan, X. Lin, F. Luo, Z. Wang, Z. Lu and Q. Zhang. Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model. ICML, 2024. https://arxiv.org/abs/2401.02051'),
      p('[3] C. K. Joshi, Q. Cappart, L.-M. Rousseau and T. Laurent. Learning TSP Requires Rethinking Generalization. CP, 2021. Data/code: https://github.com/chaitjo/learning-tsp (pinned revision in configs/sources.json).'),
      p('[4] B. Hui et al. Qwen2.5-Coder Technical Report. 2024. https://arxiv.org/abs/2409.12186'),
      p('[5] F. Luo, X. Lin, F. Liu, Q. Zhang and Z. Wang. Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization. NeurIPS, 2023. https://arxiv.org/abs/2310.07985'),
      p('Research source/version details, author asset hashes, notices and LKH mirror conditions are provided in docs/DATA_AND_MODELS.md. All reference-solver results in Table 1 were measured in this project, not copied from published papers.'),
    ]]
    return pages


def build(pages,out,fontdir):
    for label,file in [('TNR','times.ttf'),('TNR-Bold','timesbd.ttf'),('TNR-Italic','timesi.ttf'),('TNR-BoldItalic','timesbi.ttf')]:
        pdfmetrics.registerFont(TTFont(label,str(fontdir/file)))
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-BoldItalic')
    margin=2.5*cm;width,height=A4;cw=width-2*margin
    style=ParagraphStyle('body',fontName='TNR',fontSize=12,leading=13.2,spaceAfter=5)
    bold=ParagraphStyle('heading',parent=style,fontName='TNR-Bold',spaceAfter=6)
    c=canvas.Canvas(str(out/'milestone.pdf'),pagesize=A4);c.setTitle('Learning TSP Heuristic Programs from Search-State Values — Milestone');c.setAuthor('Hongye XU')
    usage=[];doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Cm(2.5)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
    normal.paragraph_format.line_spacing=1.;normal.paragraph_format.space_after=Pt(5)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman')
    for page_index,blocks in enumerate(pages):
        y=height-margin
        if page_index:doc.add_page_break()
        for block in blocks:
            kind=block['kind']
            if kind in ('p','h'):
                paragraph=Paragraph(html.escape(block['text']),bold if kind=='h' else style)
                _,hh=paragraph.wrap(cw,height);assert y-hh>=margin+15,(page_index+1,block['text'][:70],y,hh)
                paragraph.drawOn(c,margin,y-hh);y-=hh+(6 if kind=='h' else 5)
                dp=doc.add_paragraph();r=dp.add_run(block['text']);r.bold=kind=='h'
            elif kind=='table':
                cells=[[Paragraph(html.escape(str(v)),style) for v in row] for row in block['rows']]
                tb=Table(cells,colWidths=[v*cw for v in block['widths']]);tb.setStyle(TableStyle([
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
                    ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
                    ('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black)]))
                _,hh=tb.wrap(cw,height);assert y-hh>=margin+15,(page_index+1,'table overflow');tb.drawOn(c,margin,y-hh);y-=hh+5
                dt=doc.add_table(rows=0,cols=len(block['rows'][0]));dt.autofit=False
                for column,frac in zip(dt.columns,block['widths']):column.width=Cm(16*frac)
                for i,row in enumerate(block['rows']):
                    cells=dt.add_row().cells
                    for cell,text in zip(cells,row):
                        dp=cell.paragraphs[0];dp.paragraph_format.space_after=Pt(2);rr=dp.add_run(str(text));rr.bold=i==0
            else:
                path=out/'figures'/f'{block["name"]}.png';pix=fitz.Pixmap(str(path));hh=cw*pix.height/pix.width
                assert y-hh>=margin+15,(page_index+1,'figure overflow');c.drawImage(str(path),margin,y-hh,width=cw,height=hh);y-=hh+2
                caption=Paragraph(html.escape(block['caption']),style);_,hh=caption.wrap(cw,height)
                assert y-hh>=margin+15,(page_index+1,'caption overflow');caption.drawOn(c,margin,y-hh);y-=hh+5
                doc.add_picture(str(path),width=Cm(16));doc.add_paragraph(block['caption'])
        c.setFont('TNR',12);c.drawRightString(width-margin,margin+1,str(page_index+1));c.showPage()
        usage.append(dict(page=page_index+1,remaining_points=y-(margin+15)))
    c.save();doc.save(out/'milestone.docx')
    return usage


def qa(out,usage):
    pdf=fitz.open(out/'milestone.pdf');assert len(pdf)==5
    fonts=set();sizes=set();pages=[];margin=2.5*cm
    for i,page in enumerate(pdf):
        assert abs(page.rect.width-A4[0])<.1 and abs(page.rect.height-A4[1])<.1
        for b in page.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for s in line['spans']:
                    fonts.add(s['font']);sizes.add(round(s['size'],2));assert abs(s['size']-12)<.01
                    x0,y0,x1,y1=s['bbox'];assert x0>=margin-.5 and x1<=A4[0]-margin+.5 and y0>=margin-3 and y1<=A4[1]-margin+3,(i+1,s)
        text=page.get_text();assert len(text)>300
        pix=page.get_pixmap(matrix=fitz.Matrix(1.3,1.3));pix.save(out/'preview'/f'page-{i+1}.png')
        pages.append(dict(page=i+1,characters=len(text),words=len(text.split()),images=len(page.get_images())))
    assert all('TimesNewRoman' in name.replace('-','') for name in fonts),fonts
    result=dict(status='passed',paper='A4',font='Times New Roman',text_size_pt=12,margin_cm=2.5,
                body_pages=4,reference_pages=1,paragraph_line_spacing='single (PDF leading13.2pt; DOCX line_spacing1.0)',
                pdf_fonts=sorted(fonts),pdf_text_sizes=sorted(sizes),page_layout=usage,pages=pages,
                pdf_sha256=sha(out/'milestone.pdf'),docx_sha256=sha(out/'milestone.docx'),
                visual_review='rendered all five pages; manual visual inspection recorded separately')
    (out/'format_verification.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--font-directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=CODE/'reports/milestone-2026-11-19');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'preview').mkdir(exist_ok=True)
    pages=content();(a.output/'report_content.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=[]
    for i,blocks in enumerate(pages):
        md.append(f'<!-- Page {i+1} -->')
        for b in blocks:
            if b['kind'] in ('h','p'):md.append(('## ' if b['kind']=='h' else '')+b['text'])
            elif b['kind']=='table':
                for j,row in enumerate(b['rows']):
                    md.append('| '+' | '.join(map(str,row))+' |')
                    if j==0:md.append('| '+' | '.join('---' for _ in row)+' |')
            else:md.append(f'![{b["caption"]}](figures/{b["name"]}.png)')
            md.append('')
    (a.output/'milestone.md').write_text('\n'.join(md),encoding='utf-8')
    usage=build(pages,a.output,a.font_directory);qa(a.output,usage);print('Report PDF/DOCX built and format checks passed')


if __name__=='__main__':main()
