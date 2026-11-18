"""Small pure arithmetic language for generated current-edge priorities."""
import ast
import hashlib
import re
import numpy as np

ARGUMENTS=('d','p','mean_d','max_d','lam','progress','stagnation','best_ratio')
FUNCTIONS={name:getattr(np,name) for name in ('sqrt','log1p','exp','minimum','maximum','abs','clip')}
INITIAL='d / (1 + p)'


def canonical_program(response):
    text=response.strip()
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    if blocks: text=blocks[0].strip()
    tree=ast.parse(text)
    if len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef):
        fn=tree.body[0]
        if (fn.name!='priority' or fn.decorator_list or fn.returns or
            fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.defaults or
            tuple(a.arg for a in fn.args.args)!=ARGUMENTS or any(a.annotation for a in fn.args.args) or
            len(fn.body)!=1 or not isinstance(fn.body[0],ast.Return)):
            raise ValueError('Expected exact priority signature and a single return expression')
        expr=fn.body[0].value
    elif len(tree.body)==1 and isinstance(tree.body[0],ast.Expr): expr=tree.body[0].value
    else: raise ValueError('Only one arithmetic expression or priority function is permitted')
    allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,
             ast.USub,ast.UAdd,ast.Call,ast.Name,ast.Load,ast.Constant)
    nodes=list(ast.walk(expr))
    if len(nodes)>96: raise ValueError('Expression exceeds 96 AST nodes')
    for node in nodes:
        if not isinstance(node,allowed): raise ValueError('Forbidden syntax: '+type(node).__name__)
        if isinstance(node,ast.Name) and node.id not in ARGUMENTS and node.id not in FUNCTIONS:
            raise ValueError('Unknown name: '+node.id)
        if isinstance(node,ast.Constant) and (type(node.value) not in (int,float) or not np.isfinite(node.value) or abs(node.value)>100):
            raise ValueError('Only finite numeric constants with magnitude <=100')
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS or node.keywords):
            raise ValueError('Only whitelisted positional numerical calls')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            v=node.right
            if isinstance(v,ast.UnaryOp) and isinstance(v.op,(ast.USub,ast.UAdd)): v=v.operand
            if not isinstance(v,ast.Constant) or type(v.value) not in (int,float) or abs(v.value)>4:
                raise ValueError('Power exponent must be a small numeric literal')
    expression=ast.unparse(expr)
    source='def priority('+', '.join(ARGUMENTS)+'):\n    return '+expression+'\n'
    return dict(expression=expression,source=source,program_sha256=hashlib.sha256(source.encode()).hexdigest())


def compile_rule(expression):
    """Call only inside a bounded isolated evaluation worker."""
    program=canonical_program(expression)
    code=compile(program['expression'],'<generated-priority>','eval')
    def rule(d,p,s):
        values=dict(d=d,p=p,mean_d=float(d.mean()),max_d=float(d.max()),lam=s['lambda_'],
                    progress=float(np.log1p(s['penalty_events'])),stagnation=float(np.log1p(s['stagnation_sweeps'])),
                    best_ratio=s['best_length']/s['current_length'])
        with np.errstate(all='raise'):
            answer=np.asarray(eval(code,{'__builtins__':{},**FUNCTIONS},values),dtype=float)
        # A scalar is a legitimate constant-priority rule; broadcast explicitly.
        if answer.ndim==0: answer=np.full(len(d),float(answer))
        if answer.shape!=d.shape or not np.isfinite(answer).all(): raise ValueError('Nonfinite or wrong shape')
        return answer
    return rule
