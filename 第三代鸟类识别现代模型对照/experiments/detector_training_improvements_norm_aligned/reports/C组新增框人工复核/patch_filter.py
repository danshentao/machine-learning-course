from pathlib import Path
p=Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照\experiments\detector_training_improvements_norm_aligned\reports\C组新增框人工复核\build_review_page.py')
s=p.read_text(encoding='utf-8-sig')
old="document.getElementById('search').oninput=e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('.section.active article').forEach(a=>a.classList.toggle('hidden',!(`${a.dataset.filename} ${a.dataset.folder}`.includes(q)))};"
new="document.getElementById('search').oninput=e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('.section.active article').forEach(a=>{const text=`${a.dataset.filename} ${a.dataset.folder}`;a.classList.toggle('hidden',!text.includes(q));});};"
assert old in s
p.write_text(s.replace(old,new),encoding='utf-8')
