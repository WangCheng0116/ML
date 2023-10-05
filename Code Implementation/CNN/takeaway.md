* 两个heuristic不一定非得一个dominate另一个, 可以没关系
* BFS is better when target is closer to Source. DFS is better when target is far from source.
* It only makes sense to discuss dominance where both heuristics are admissible.
* 算entropy取最小, IG选最大哦
* consistency -> admissiblity, not admissble -> not consistent
* An optimal solution (minimum number of steps) can always be found if we employ the
right search algorithm. 但是答案可能不存在哦, 所以不能选
goal state 不知道的时候goal state是满足goal test的
* 说的越多越好, 尽量详细, 约束条件能写就写, 比如烧杯液体体积小于容量
* admissible 得h都大于等于0
* reflection中的三个学习的地方
1. pruning can speed up algorithm without affecting results
2. sometimes pruning in decision tree can avoid overfitting
3. A better heuristic is more likely to result in a better performance