"""Standard-library check of summary math, completeness and failure handling."""
from review_recovery import summarize, PAIRS, TRUTHS

rows = []
for t in TRUTHS:
    for i in (1, 2):
        row = dict(truth=t, SNR='3', repetition=str(i), seed=str(7+10000+i), success='1')
        for pair in PAIRS:
            for metric in ('FF', 'FB', 'H'):
                v = {'observed':1., 'null':0., 'reversed':-1.}[t]
                row['true_'+pair+'_'+metric] = str(v)
                row['estimated_'+pair+'_'+metric] = str(v+(-.1 if i==1 else .1))
        rows.append(row)
s = summarize(rows, [3], 2, 7)
assert len(s)==27 and abs(s[0]['rmse']-.1)<1e-12
assert s[0]['sign_recovery_fraction']==1 and s[9]['sign_recovery_fraction']==''
rows[0]['success']='0'
s = summarize(rows, [3], 2, 7)
assert s[0]['failed']==1 and s[0]['summary_scope'].startswith('INCOMPLETE')
for bad in (rows[:-1], rows+[rows[0]]):
    try: summarize(bad, [3], 2, 7)
    except ValueError: pass
    else: raise AssertionError('Incomplete/duplicate grid accepted')
print('PASS: summary math, null signs, retained failures, missing/duplicate cases')

# Regression guard for the server-reported MATLAB struct-assignment failure.
# This is a source check, not execution of MATLAB; server selftest is required.
from pathlib import Path
matlab = Path(__file__).with_name('recovery_nine.m').read_text()
assert 'struct([])' not in matlab
assert "truths=repmat(struct('name','','A',zeros(9),'max_real_eigenvalue',NaN),3,1);" in matlab
assert 'source_rows{i}=row;' in matlab and 'rows=vertcat(source_rows{:});' in matlab
print('PASS: MATLAB allocation source guard (MATLAB execution still pending)')

# Only one conversion implementation; all callers use its scalar-safe contract.
assert matlab.count('struct2table(')==1
assert "struct2table(records(:),'AsArray',true)" in matlab
assert "one=records_table(record)" in matlab
assert "two=records_table([record;failed])" in matlab
assert "readtable(file,'TextType','string')" in matlab
print('PASS: centralized scalar-safe table conversion source guard; MATLAB roundtrip pending server')
