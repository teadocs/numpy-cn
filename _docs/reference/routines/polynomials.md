# 多项式

NumPy中的多项式可以使用NumPy 1.4中引入的``numpy.polynomial``包的使用方便类来创建，操作甚至拟合。

在NumPy 1.4之前，``numpy.poly1d``是首选类，它仍然可用以保持向后兼容性。 但是，较新的Polynomial包比``numpy.poly1d``更完整，并且它的便利类在numpy环境中表现得更好。 因此，建议使用多项式进行新的编码。

## 过渡通知

Polynomial包中的各种例程都处理系列，其系数从零度向上，这是Poly1d约定的逆序。 记住这一点的简单方法是索引对应于度，即，coef[i]是度i的项的系数。