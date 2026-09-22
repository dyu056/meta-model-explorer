"""Label-free graph features; control scopes are connected by explicit boundary edges."""
from dataclasses import dataclass
import math
import torch
from ..graph_ir.analysis import FEATURE_TYPES
from ..graph_ir.validation import split_ref

PORTS=('x','other','weight','weights','q','k','scores','mask','logits','values','xs','state','fragment','carry','y','out','capture','binding','body_output','self')
ACTIVATIONS=('relu','gelu','silu','sigmoid','tanh','identity')
REDUCERS=('sum','mean','max')
NUMERIC=('rank','numel','dim0','dim1','dim2','dim3','stored_depth','scope_depth','steps','shared','per_step','axis','offset','trainable','parameter_numel')
FEATURE_COLUMNS=tuple(FEATURE_TYPES)+NUMERIC+tuple('activation:'+s for s in ACTIVATIONS)+tuple('reducer:'+s for s in REDUCERS)


@dataclass
class EncodedGraph:
    x:torch.Tensor
    allowed:torch.Tensor
    relations:torch.Tensor
    distances:torch.Tensor
    statistics:torch.Tensor
    ids:list[str]


def encode(graph,batch_size=64):
    # Explorer STEP 01: 确定性编码：节点属性与有向图关系。
    # LaTeX: $$G \mapsto (X_{raw},E_{raw},M_{raw},D_{raw}),\quad X_{raw}\in\mathbb{R}^{N\times46}$$
    exported=graph.matrices({'B':batch_size})
    scopes=exported['scopes'];keys=[];rows=[];mapping={};stats=[0.]*len(FEATURE_TYPES)
    for scope,data in sorted(scopes.items()):
        definitions={n['id']:n for n in data['nodes']}
        for i,(name,kind) in enumerate(zip(data['node_ids'],data['node_types'])):
            mapping[(scope,name)]=len(rows);keys.append(scope+'/'+name)
            attrs=definitions.get(name,{}).get('attrs',{})
            specs=[v for k,v in data['tensor_specs'].items() if k.startswith(name+':')]
            if not specs:
                specs=[e['tensor'] for e in data['edges'] if e['target']==name]
            shape=specs[0]['shape'] if specs else []
            dims=[math.log1p(v)/10 for v in shape[:4]]+[0.]*(4-len(shape[:4]))
            param=data['parameters'].get(name,{})
            pnum=math.prod(param['shape']) if param else 0
            numeric=[len(shape)/4,math.log1p(math.prod(shape))/10,*dims,
                     data['stored_depth'][i]/20,scope.count('/')/4,math.log1p(attrs.get('steps',0))/4,
                     float(attrs.get('binding')=='shared'),float(attrs.get('binding')=='per_step'),
                     float(attrs.get('dim',0)) if isinstance(attrs.get('dim',0),int) else 0.,float(attrs.get('offset',0)),
                     float(param.get('trainable',False)),math.log1p(pnum)/10]
            row=data['node_features'][i]+numeric+[float(attrs.get('activation')==a) for a in ACTIVATIONS]+[float(attrs.get('reducer')==a) for a in REDUCERS]
            rows.append(row);stats[FEATURE_TYPES.index(kind)]+=1
    n=len(rows);relations=torch.zeros(n,n,len(PORTS));direct=torch.zeros(n,n,dtype=torch.bool)
    def add(source,target,port):
        direct[target,source]=True
        relations[target,source,PORTS.index(port if port in PORTS else 'capture')]+=1
    for scope,data in scopes.items():
        for edge in data['edges']:add(mapping[(scope,edge['source'])],mapping[(scope,edge['target'])],edge['target_port'])
    for relation in exported['hierarchy']:
        parent=relation['parent_scope'];child=relation['body_scope'];control=relation['control_node']
        # Hierarchical input bindings preserve which external tensor is read by the body.
        for port,refs in relation['bindings'].items():
            target_port='fragment' if port=='xs' else port
            if (child,target_port) not in mapping:continue
            for ref in refs if isinstance(refs,list) else [refs]:
                name,_=split_ref(ref)
                add(mapping[(parent,name)],mapping[(child,target_port)],'binding')
        for name,kind in zip(scopes[child]['node_ids'],scopes[child]['node_types']):
            if kind=='OUTPUT':add(mapping[(child,name)],mapping[(parent,control)],'body_output')
    distance=torch.full((n,n),n+1,dtype=torch.long);distance[direct]=1;distance.fill_diagonal_(0)
    for k in range(n):distance=torch.minimum(distance,distance[:,k,None]+distance[k,None,:])
    allowed=distance<=n
    relations[torch.arange(n),torch.arange(n),PORTS.index('self')]=1
    distances=distance.clamp(max=8)
    pnum=sum(math.prod(p.shape) for p in graph.parameters.values() if p.trainable)
    summary=[math.log1p(v) for v in stats]+[math.log1p(pnum),math.log1p(n),float(max(d['stored_depth'][-1] for d in scopes.values()))/20]
    return EncodedGraph(torch.tensor(rows,dtype=torch.float32),allowed,relations,distances,torch.tensor(summary,dtype=torch.float32),keys)


def batch_graphs(graphs,task_ids=None):
    """Pad graphs; omit task_ids for one-model-per-task predictors.

    Explicit task_ids retain the legacy task field for older predictors.
    """
    if not graphs or (task_ids is not None and len(graphs)!=len(task_ids)):
        raise ValueError('Need nonempty graphs and matching task_ids when supplied')
    size=max(len(g.ids) for g in graphs);b=len(graphs)
    x=torch.zeros(b,size,len(FEATURE_COLUMNS));valid=torch.zeros(b,size,dtype=torch.bool)
    mask=torch.eye(size,dtype=torch.bool).expand(b,-1,-1).clone()
    relations=torch.zeros(b,size,size,len(PORTS));dist=torch.full((b,size,size),8,dtype=torch.long)
    for i,g in enumerate(graphs):
        n=len(g.ids);x[i,:n]=g.x;valid[i,:n]=True;mask[i,:n,:n]=g.allowed
        relations[i,:n,:n]=g.relations;dist[i,:n,:n]=g.distances
    batch={'x':x,'valid':valid,'allowed':mask,'relations':relations,'distances':dist,
           'statistics':torch.stack([g.statistics for g in graphs])}
    if task_ids is not None:
        batch['task']=torch.tensor(task_ids,dtype=torch.long)
    return batch
