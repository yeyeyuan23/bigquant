from pathlib import Path
import json
import pandas as pd
ROOT = Path('.')
top = pd.read_csv('reports/runtime_all156_full_20260729/routes/tree_lightgbm_importance_selection.csv').sort_values('importance_rank').head(50)
top_ids = [x.replace('self__','') for x in top.feature]
# placeholder generator smoke: records chosen top50 and creates notebook skeleton
source = '''"""Top50 T-route rolling LightGBM submission v01."""

# TODO generated full implementation pending stable remote sync.
TOP50 = ''' + repr(top_ids) + '''
'''
(ROOT/'submissions/lgbm_t_top50_v01.py').write_text(source, encoding='utf-8')
nb = {'cells':[{'cell_type':'markdown','id':'factor-description','metadata':{},'source':['# BigAlpha 2026 T importance top50 LightGBM v01\n','placeholder']},{'cell_type':'code','execution_count':None,'id':'factor-code','metadata':{},'outputs':[],'source':source.splitlines(True)}], 'metadata': {'kernelspec': {'display_name':'Python 3.11.8','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3.11'}}, 'nbformat':4, 'nbformat_minor':5}
(ROOT/'submissions/lgbm_t_top50_v01.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print('wrote placeholder top50', len(top_ids))
