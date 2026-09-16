import csv
import tempfile
from pathlib import Path
from evaluate import evaluate, EvalConfig, dump_summary_csv, format_summary
from detection import Instance

def inst(i, text='甲', ignore=False):
    return Instance(region=[i*20, 0, i*20+10, 10], text=text, ignore=ignore)
def close(a,b):
    assert abs(a-b)<1e-10,(a,b)

def main():
    gt={'a':[inst(i,'甲'*25) for i in range(5)]}
    pred={'a':[inst(i,'甲'*23+'乙乙') for i in range(4)]}
    r=evaluate(gt,pred)
    close(r['detection']['recall'],.8)
    close(r['one_ned_details']['matched_only']['score'],.92)
    close(r['one_ned_details']['gt_penalized']['score'],.736)
    close(r['one_ned_details']['full_penalty']['score'],.736)
    r=evaluate({'a':[inst(0)]},{'a':[inst(0),inst(1)]})
    close(r['one_ned_details']['gt_penalized']['score'],1)
    close(r['one_ned_details']['full_penalty']['score'],.5)
    g={'a':[inst(0),Instance(region=[0,0,9,10],text='',ignore=True)]}
    p={'a':[Instance(region=[0,0,9,10],text='甲')]}
    for strategy in ('greedy','cardinality_iou'):
        close(evaluate(g,p,EvalConfig(matching=strategy))['counts']['tp'],1)
        close(evaluate(g,p,EvalConfig(matching=strategy,ignore_priority=False))['counts']['tp'],0)
    r=evaluate({'a':[inst(0,'',True)]},{'a':[inst(0)]})
    assert r['counts']['fp']==0
    assert r['one_ned_details']['full_penalty']['score'] is None
    g={'a':[inst(i) for i in range(3)],'b':[inst(0)]}
    p={'a':[inst(i) for i in range(3)],'b':[inst(0,'乙')]}
    r=evaluate(g,p)
    close(r['one_ned_details']['matched_only']['score'],.75)
    close(r['one_ned_macro_details']['matched_only']['score'],.5)
    r=evaluate({'a':[inst(0)]},{'a':[]})
    assert r['one_ned_details']['matched_only']['score'] is None
    close(r['one_ned_details']['full_penalty']['score'],0)
    r=evaluate({'a':[]},{'a':[]})
    assert all(v['score'] is None for v in r['one_ned_details'].values())
    assert r['one_ned_macro_details']['matched_only']['excluded_images']==1
    assert 'N/A' in format_summary(r)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'summary.csv';dump_summary_csv(r,str(path),'toy','test')
        rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        assert rows[0]['one_ned_matched_only_micro_percent']==''
    print('Protocol regression cases passed')

if __name__=='__main__':main()
