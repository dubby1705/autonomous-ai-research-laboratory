import re

path = 'Engine/Question_Engine/Mathematics.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add domain param to deepen_cluster
old1 = 'def deepen_cluster(hypothesis: str, cluster_docs: List[str], num_new: int = 10) -> List[MathIdea]:'
new1 = 'def deepen_cluster(hypothesis: str, cluster_docs: List[str], num_new: int = 10, domain: str = "general") -> List[MathIdea]:'
if old1 in content:
    content = content.replace(old1, new1)
    print('Fix 1: domain param added')
else:
    print('Fix 1 FAILED')

# Fix 2: Pass domain when calling deepen_cluster
old2 = 'deepened_ideas = deepen_cluster(hypothesis, cinfo["documents"], num_new=8)'
new2 = 'deepened_ideas = deepen_cluster(hypothesis, cinfo["documents"], num_new=8, domain=math_domain)'
if old2 in content:
    content = content.replace(old2, new2)
    print('Fix 2: domain passed in call')
else:
    print('Fix 2 FAILED')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')