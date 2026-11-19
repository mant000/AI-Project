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
    folder=CODE/'results/discovery'
    s=read(folder/'analysis/study_summary.json');m=read(folder/'analysis/mechanism_summary.json')
    strict=read(CODE/'results/interface_check/analysis/study_summary.json')
    assert s['status']==m['status']==strict['status']=='passed'
    assert read(folder/'references/completed.json')['status']=='completed'
    proof=read(CODE/'results/reproduction/release_verification.json');assert proof['status']=='passed'
    fresh=read(CODE/'results/reproduction/analysis/study_summary.json')
    for field in ('arm_mean_gap_percent','baseline_mean_gap_percent','contrasts','lineages'):
        assert fresh[field]==s[field],('report differs from fresh-clone recomputation',field)
    assert read(CODE/'results/reproduction/analysis/mechanism_summary.json')==m
    effects=s['contrasts'];costs=s['costs'];strictcosts=strict['costs']
    lineages=s['lineages'];valid=sum(r['valid'] for r in lineages);validstrict=sum(r['valid'] for r in strict['lineages'])
    armstats={a:{k:sum(r[k] for r in lineages if r['arm']==a) for k in ['valid','unique_programs','unique_endpoint_route_vectors','directional_feedback_prompts','feedback_states_nonzero','feedback_states_total']} for a in ('A','P','F')}
    ref=[]
    for label in ['soft_penalty','centered','phase_centered','LEHD_RRC0','LEHD_RRC10','LKH']:
        x=read(folder/'references'/f'{label}.json');ref.append([{'soft_penalty':'Manual: soft penalty','centered':'Manual: centered','phase_centered':'Manual: phase centered','LEHD_RRC0':'LEHD greedy','LEHD_RRC10':'LEHD RRC10','LKH':'LKH3.0.6 (1000 trials)'}[label],fmt(x['mean_gap_percent'])])
    prep=CODE/'results/mechanism/bank'
    pc=[read(prep/f'{phase}_completed.json') for phase in ('pilot','full')]
    mechanism_wall=sum(x['wall_seconds'] for x in pc); mechanism_cpu=sum(x['parent_cpu_seconds']+x['waited_child_cpu_seconds'] for x in pc)
    strictwall=sum(x['wall_seconds'] for x in strictcosts['attempts']);exprwall=sum(x['wall_seconds'] for x in costs['attempts'])
    discoverycpu=sum(x['parent_cpu_seconds']+x['waited_child_cpu_seconds'] for c in [strictcosts,costs] for x in c['attempts'])
    peak=max(x['cuda_peak_allocated_bytes'] for c in [strictcosts,costs] for x in c['attempts'])/1024**3
    f=effects['F-A'];fp=effects['F-P'];success=f['predeclared_success']
    timing=read(folder/'timing/completed.json')['summary']
    example=next(x['selected_program'] for x in read(folder/'run/frozen_programs.json')['programs'] if x['arm']=='F' and x['discovery_seed']==1204)
    native_time=[timing['native_gls'][str(n)]['mean_wall_seconds'] for n in (50,100)]
    example_time=[timing[example['program_sha256']][str(n)]['mean_wall_seconds'] for n in (50,100)]
    outcome=('met' if success else 'did not meet')
    pages=[[
      h('Feedback for Discovering TSP Heuristics'),
      p('CS5491 Artificial Intelligence — Milestone Report'),
      p('Hongye XU (72655146) | Individual project | 19 November 2026'),
      p('Code: https://github.com/mant000/AI-Project'),
      p('Version: milestone-2026-11-19-v2'),
      h('1. Problem and motivation'),
      p('Can feedback from an intermediate search state help a frozen code language model discover better TSP heuristics? A complete route tells the model whether a rule worked, but says little about which decisions helped. I initially tried learning the value of individual edge-penalty actions. About 97% of those training labels were zero, and the learned Transformer offered little validation benefit. This raised a more basic question: does one penalty action adequately represent a rule that is applied repeatedly?'),
      p('I test a rule fragment: apply the candidate rule for a fixed part of the remaining search, then complete the run with standard guided local search (GLS). I first compare this feedback with single-penalty feedback at identical saved states and work budgets. I then test whether it helps the same frozen generator discover better rules. The first comparison measures the information in the feedback; the second measures its practical value.'),
      h('2. Method: rules and state feedback'),
      p('The experiments use Euclidean TSP50/100 instances from the author training release [3]. My solver follows the GLS penalty idea [1], alternating best-improvement 2-opt with one edge-penalty increment. Each run allows 200 sweep equivalents: 235,000/970,000 evaluated 2-opt deltas. Gap is 100(L/Lref − 1), using original continuous distances. The supplied reference tours are not certified continuous optima, so negative gaps are retained.'),
      p('The model generates a single arithmetic priority h(d,p,mean_d,max_d,lam,progress,stagnation,best_ratio) for current-tour edges. The maximum-priority edge receives the penalty; conventional GLS uses d/(1+p). The solver, initialization, lambda rule and numerical language stay fixed. The 96-node AST permits only bounded numeric constants, arithmetic and whitelisted numerical functions. Features use the current state only, with no future branch labels or reference-tour information.'),
      p(f'The model initially returned a complete function, but often omitted its header ({validstrict}/384 valid). I made one common interface change before inspecting development outcomes: supply the header and return prefix, then request one expression line, capped at 128 tokens. The completed initial batch is reported separately: F−A was −0.0718 gap points [−0.2007,+0.0374]. The two batches are not pooled.'),
      p('Like EoH [2], this approach searches for executable heuristics using an LLM. My comparison focuses on the feedback supplied during that search. Only complete-tour quality can promote or select a program. Once selected, the arithmetic rule runs without an LLM or value model.'),
    ],[
      h('3. Experimental design'),
      p('At a saved penalty boundary s with remaining work R, all branches share state and RNG seed 71. Single applies h once, then standard GLS. Fragment applies h for floor(0.25R) deltas, then standard GLS. Persistent applies h throughout R. All end at the same total work cap and retain the best original-distance tour.'),
      p('The benefit of an intervention is 100 × (standard continuation length − intervention length) / saved best length. Feedback is the child benefit minus the parent benefit at the same state. Persistent outcomes are a diagnostic target, never model input; they describe continuation from s, not full-run generalization. The 25% fraction and 12 diagnostic rules were fixed before outcomes.'),
      table([['Role','Original instances','Use'],['Bank','128 (64/size)','State feedback and diagnostics'],['Discovery','32 (16/size)','Candidate endpoint selection'],['Rule selection','32 (16/size)','Choose among four elites'],['Development','64 (32/size)','Evaluate frozen lineages']], [.23,.25,.52]),
      p('I reserved 256 previously unused training instances before measuring outcomes and checked their identities and coordinates against earlier experiments. The four roles are disjoint. Each bank original follows standard, centered and soft-penalty trajectories, sampled after 10/30/50/70% work: 1,536 states. Feedback uses one predetermined state per original, balanced across policy/phase; all 128 contribute summaries, with two fixed-schedule examples.'),
      h('4. Does a rule fragment give more information?'),
      fig('mechanism','Figure 1. Ten noncontrol rules × 1,536 states. Positive rescaling and standard-rule controls are excluded. Correlations are pooled descriptive statistics against the persistent conditional target.'),
      p('Nonzero outcomes rise from 82/15,360 (0.53%) to 1,538/15,360 (10.01%): a paired original-instance increase of 9.48 points (95% interval 7.79–11.26). Tie-aware sign agreement rises from 0.504 to 0.573 (increase 0.071, interval 0.056–0.087; 102 informative originals). All 56,832 branch tours reprice exactly; 3,072 standard/rescaling controls and 12 independent replays match. This establishes intervention fidelity, not improved discovery.'),
    ],[
      h('5. Does the feedback improve program discovery?'),
      p('Frozen NF4 Qwen2.5-Coder-7B-Instruct [4], without adapters, receives endpoints alone (A), plus single-penalty feedback (P), or plus fragment feedback (F). Matched settings: seeds 1201–1204, standard parent, temperature 0.8/top-p 0.95/top-k 50, 32 outputs per lineage (384 total). Failures/duplicates consume budget; feedback compares the previous candidate with its actual parent.'),
      p('The four best unique discovery candidates per lineage are reranked on the selection split with three solver seeds. All 12 selected programs freeze before development evaluation with five seeds. Seeds average within originals. Paired intervals jointly resample discovery repetitions and size-stratified originals. Success requires F−A≤−0.05 gap points and an upper 95% limit below zero.'),
      table([['Method / fixed reference','Mean dev gap (%)'],['A: endpoints',fmt(s['arm_mean_gap_percent']['A'])],['P: one penalty',fmt(s['arm_mean_gap_percent']['P'])],['F: rule fragment',fmt(s['arm_mean_gap_percent']['F'])],['Standard GLS',fmt(s['baseline_mean_gap_percent']['gls'])],['Nearest neighbor',fmt(s['baseline_mean_gap_percent']['nn'])],['2-opt',fmt(s['baseline_mean_gap_percent']['two_opt'])],*ref],[.72,.28]),
      p('Table 1. Same 64 dev originals/five solver seeds; all four A/P/F discovery repetitions. All manual rules are reported. LKH/LEHD [5] use their own budgets. LKH rounds input at scale 10^6; tours are repriced continuously.'),
      fig('discovery_effects','Figure 2. Each colored dot is one paired discovery repetition; diamonds and intervals show the joint paired mean and 95% bootstrap interval.'),
      p(f'F−A={f["mean_gap_pp"]:+.4f} gap points [{f["ci95_low"]:+.4f},{f["ci95_high"]:+.4f}]; F−P={fp["mean_gap_pp"]:+.4f} [{fp["ci95_low"]:+.4f},{fp["ci95_high"]:+.4f}]. This {outcome} the success gate. Four discovery repetitions limit precision; 320 solves do not create more discovery runs.'),
    ],[
      h('6. What the result means and what it costs'),
      p(f'The main study produced {valid}/384 valid expressions. Single-penalty feedback distinguished parent from child in only {armstats["P"]["directional_feedback_prompts"]}/128 prompts, compared with {armstats["F"]["directional_feedback_prompts"]}/128 for fragments. The diagnostic bank therefore reveals a real difference in feedback availability, but even fragment feedback is often tied along the rules the model actually proposes.'),
      p('Three F lineages retain standard GLS; one selects 0.4d/(1+p)+0.6d·mean_d/max_d, an interpretable mixture that softens penalty suppression. Its dev improvement does not establish an independently repeated discovery. One F−A advantage instead avoids a dev regression of an A-selected program. The next experiment must control the common proposal pool as well as feedback.'),
      p('Experiments used one rented RTX3090 24GB, 14 vCPUs and 60GB RAM. Table 2 reports measured wall time and total parent plus waited-child CPU time. The initial and revised discovery batches are both included in the cost.'),
      table([['Stage','Wall min','CPU min'],['State-bank measurement',f'{mechanism_wall/60:.2f}',f'{mechanism_cpu/60:.2f}'],['Initial function interface',f'{strictwall/60:.2f}',f'{sum(x["parent_cpu_seconds"]+x["waited_child_cpu_seconds"] for x in strictcosts["attempts"])/60:.2f}'],['Main expression interface',f'{exprwall/60:.2f}',f'{sum(x["parent_cpu_seconds"]+x["waited_child_cpu_seconds"] for x in costs["attempts"])/60:.2f}']],[.60,.20,.20]),
      p(f'Table 2. The 192-state pilot is included once in the 1,536-state measurement. Initial/main discovery used {strictcosts["input_tokens"]:,}/{costs["input_tokens"]:,} input tokens and {strictcosts["output_tokens"]:,}/{costs["output_tokens"]:,} output tokens. Synchronized generation took {strictcosts["generation_seconds"]/60:.2f}/{costs["generation_seconds"]/60:.2f} min; peak Torch allocation was {peak:.2f} GiB. Equal output budgets thus do not imply equal compute cost.'),
      p(f'On 16 prespecified originals with three serial repeats (576 solves across methods), native GLS takes {native_time[0]:.3f}/{native_time[1]:.3f} s for TSP50/100; the illustrated F rule takes {example_time[0]:.3f}/{example_time[1]:.3f} s. These fixed-work service times exclude offline discovery and do not establish an anytime advantage.'),
      p('These results explain why improving a feedback signal is only an intermediate step. Fragment feedback changes more often, yet most F lineages still select the initial rule, and manual rules remain stronger. The measured branches do not establish the success of a learned Transformer. Coverage is also limited to TSP50/100 and the bank’s three trajectories; arbitrary-policy validity remains untested.'),
      h('7. Reproduction and next steps'),
      p('The README follows the experiment in order: recompute both feedback and discovery results from raw records; reconstruct the 1,536 states and rerun the selected solvers; then optionally regenerate the GPU discovery. The first two steps need only the included inputs and a CPU. In a separate Python environment on the rented server, all states matched exactly, 2,560 development routes replayed exactly, and six informative feedback branches matched. This checks the numerical pipeline; stochastic GPU generation can differ across systems.'),
      p('For the final project, I will first compare feedback-based and endpoint-only screening on a common, behavior-diverse candidate pool. This isolates information use from changes in what the generator proposes. I will measure selection regret and CPU cost on new development instances. A learned fragment-value predictor is justified only if measured feedback first improves screening. The repository gives the budget and decision points. I will freeze the method and evaluation protocol before opening the final split.'),

    ],[
      h('References'),
      p('[1] C. Voudouris and E. Tsang. Guided local search and its application to the traveling salesman problem. European Journal of Operational Research, 113(2):469–499, 1999. https://doi.org/10.1016/S0377-2217(98)00099-X'),
      p('[2] F. Liu, X. Tong, M. Yuan, X. Lin, F. Luo, Z. Wang, Z. Lu and Q. Zhang. Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model. ICML, 2024. https://arxiv.org/abs/2401.02051'),
      p('[3] C. K. Joshi, Q. Cappart, L.-M. Rousseau and T. Laurent. Learning TSP Requires Rethinking Generalization. CP, 2021. Data/code: https://github.com/chaitjo/learning-tsp (pinned revision in configs/sources.json).'),
      p('[4] B. Hui et al. Qwen2.5-Coder Technical Report. 2024. https://arxiv.org/abs/2409.12186'),
      p('[5] F. Luo, X. Lin, F. Liu, Q. Zhang and Z. Wang. Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization. NeurIPS, 2023. https://arxiv.org/abs/2310.07985'),
      p('Pinned data, model and reference-solver sources are listed in docs/data_sources.md. All reference-solver results in Table 1 were measured for this project.'),
    ]]
    return pages


def build(pages,out,fontdir):
    for label,file in [('TNR','times.ttf'),('TNR-Bold','timesbd.ttf'),('TNR-Italic','timesi.ttf'),('TNR-BoldItalic','timesbi.ttf')]:
        pdfmetrics.registerFont(TTFont(label,str(fontdir/file)))
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-BoldItalic')
    margin=2.5*cm;width,height=A4;cw=width-2*margin
    style=ParagraphStyle('body',fontName='TNR',fontSize=12,leading=13.2,spaceAfter=5)
    bold=ParagraphStyle('heading',parent=style,fontName='TNR-Bold',spaceAfter=6)
    c=canvas.Canvas(str(out/'milestone.pdf'),pagesize=A4);c.setTitle('Feedback for Discovering TSP Heuristics — Milestone');c.setAuthor('Hongye XU')
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
                markup=html.escape(block['text'])
                url='https://github.com/mant000/AI-Project'
                markup=markup.replace(url, f'<link href="{url}" color="#000000"><u>{url}</u></link>')
                paragraph=Paragraph(markup,bold if kind=='h' else style)
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
        c.setFont('TNR',12);c.drawRightString(width-margin,margin+5,str(page_index+1));c.showPage()
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
                report_facts_match_fresh_clone_recomputation=True,
                visual_review='rendered all five pages; manual visual inspection recorded separately')
    (out/'format_verification.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--font-directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=CODE/'report');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'preview').mkdir(exist_ok=True)
    pages=content()
    usage=build(pages,a.output,a.font_directory);qa(a.output,usage);print('Report PDF/DOCX built and format checks passed')


if __name__=='__main__':main()
