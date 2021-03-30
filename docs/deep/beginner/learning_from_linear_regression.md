---
meta:
  - name: keywords
    content: 线性回归,MNIST,手写数字数据集
  - name: description
    content: 本教程的话题将围绕着“数据（Data）”进行，希望你能够有所收获。我们已经学习了如何去训练一个简单的线性回归模型，接下来我们将稍微升级一下难度。
---

# 一个稍微复杂些的线性回归模型

本文章原文链接：[https://megengine.org.cn/doc/stable/zh/getting-started/beginner/learning-from-linear-regression.html](https://megengine.org.cn/doc/stable/zh/getting-started/beginner/learning-from-linear-regression.html?id=NumpyLinear_page2_docstart_01)

我们已经学习了如何去训练一个简单的线性回归模型 $y = w * x + b$, 接下来我们将稍微升级一下难度：

- 线性回归由一元变为多元，我们对单个样本数据的表示形式将不再是简单的标量 $x$, 而是升级为多维的向量 $\mathbf {x}$ 表示；
- 我们将接触到更多不同形式的梯度下降策略，从而接触一个新的超参数 `batch_size`;
- 我们将尝试将数据集封装成 MegEngine 支持的 `Dataset` 类，方便在 `Dataloader` 中进行各种预处理操作；
- 同时，我们的前向计算过程将变成矩阵运算形式，能够帮助你更好地理解向量化实现的优点，以及由框架完成求梯度操作的好处；
- 在这个过程中，一些遗留问题会得到解答，同时我们将接触到一些新的机器学习概念。

本教程的话题将围绕着 **“数据（Data）”** 进行，希望你能够有所收获～

请先运行下面的代码，检验你的环境中是否已经安装好 MegEngine（[访问官网安装教程](https://megengine.org.cn/install/?id=NumpyLinear_page2_install_01)）：

```python
import megengine

print(megengine.__version__)
```

输出结果：

```
1.2.0
```

接下来，我们将对将要使用的数据集 [波士顿房价数据集](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) 进行一定的了解。

或者你可以可前往 **MegStudio**，fork 公开项目，无需本地安装，直接线上体验（[前往学习](https://studio.brainpp.com/project/4642/?id=NumpyLinear_page2_studio_01)）

## 原始数据集的获取和分析

获取到真实的数据集原始文件后，往往需要做大量的数据格式处理工作，变成易于计算机理解的形式，才能被各种框架和库使用。

但目前我们的重心不在此处，使用 Python 机器学习库 [scikit-learn](https://scikit-learn.org/stable/index.html)，可以快速地获得波士顿房价数据集:

```python
import numpy as np
from sklearn.datasets import load_boston

# 获取 boston 房价数据集，需要安装有 scikit-learn 这个库
boston = load_boston()

# 查看 boston 对象内部的组成结构
print(boston.keys())

# 查看数据集描述，内容比较多，因此这里注释掉了
# print(boston.DESCR)

# 查看 data 和 target 信息，这里的 target 可以理解成 label
print(boston.data.shape, boston.target.shape)
```

输出结果：

```
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
(506, 13) (506,)
```

我们已经认识过了数据 `data` 和标签 `label`（标签有时也叫目标 `target` ），那么 `feature` 是什么？

我们在描述一个事物的时候，通常会寻找其属性（Attribute）或者说特征（Feature）：

- 比如我们描述一个人的长相，会说这个人的鼻子如何、眼睛如何、额头如何等等
- 又比如在游戏角色的属性经常有生命值、魔法值、攻击力、防御力等属性
- 类比到一些支持自定义角色的游戏，这些特征就变成许多可量化和调整的数据

实际上，你并不需要在意波士顿房价数据集中选用了哪些特征，我们在本教程中关注的更多是“特征维度变多”这一情况。

为了方便后续的交流，我们需要引入一些数学符号来描述它们，不用害怕，理解起来非常简单：

- 波士顿房价数据集的样本容量为 506，记为 $n$; （Number）
- 每个样本的特征有 13 个维度，包括住宅平均房间数、城镇师生比例等等，记为 $d$. （Dimensionality）
- 单个的样本可以用一个向量  $\mathbf {x}=(x_{1}, x_{2}, \ldots , x_{d})$ 来表示，称为“特征向量”，里面的每个元素对应着该样本的某一维特征；
- 数据集 `data` 由 $n$ 行特征向量组成，每个特征向量有 $d$ 维度特征，因此整个数据集可以用一个形状为 $(n, d)$ 的数据矩阵 $X$ 来表示；
- 标签 `label` 的每个元素是一个标量值，记录着房价值，因此它本身是一个含有 $n$ 个元素的标签向量 $\mathbf {y}$.

<svg xmlns="http://www.w3.org/2000/svg" width="60.955ex" height="13.261ex" role="img" focusable="false" viewBox="0 -3180.8 26942.3 5861.5" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" style="vertical-align: -6.065ex;"><defs><path id="MJX-13-TEX-I-1D44B" d="M42 0H40Q26 0 26 11Q26 15 29 27Q33 41 36 43T55 46Q141 49 190 98Q200 108 306 224T411 342Q302 620 297 625Q288 636 234 637H206Q200 643 200 645T202 664Q206 677 212 683H226Q260 681 347 681Q380 681 408 681T453 682T473 682Q490 682 490 671Q490 670 488 658Q484 643 481 640T465 637Q434 634 411 620L488 426L541 485Q646 598 646 610Q646 628 622 635Q617 635 609 637Q594 637 594 648Q594 650 596 664Q600 677 606 683H618Q619 683 643 683T697 681T738 680Q828 680 837 683H845Q852 676 852 672Q850 647 840 637H824Q790 636 763 628T722 611T698 593L687 584Q687 585 592 480L505 384Q505 383 536 304T601 142T638 56Q648 47 699 46Q734 46 734 37Q734 35 732 23Q728 7 725 4T711 1Q708 1 678 1T589 2Q528 2 496 2T461 1Q444 1 444 10Q444 11 446 25Q448 35 450 39T455 44T464 46T480 47T506 54Q523 62 523 64Q522 64 476 181L429 299Q241 95 236 84Q232 76 232 72Q232 53 261 47Q262 47 267 47T273 46Q276 46 277 46T280 45T283 42T284 35Q284 26 282 19Q279 6 276 4T261 1Q258 1 243 1T201 2T142 2Q64 2 42 0Z"></path><path id="MJX-13-TEX-N-3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path><path id="MJX-13-TEX-S4-23A1" d="M319 -645V1154H666V1070H403V-645H319Z"></path><path id="MJX-13-TEX-S4-23A3" d="M319 -644V1155H403V-560H666V-644H319Z"></path><path id="MJX-13-TEX-S4-23A2" d="M319 0V602H403V0H319Z"></path><path id="MJX-13-TEX-N-2212" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"></path><path id="MJX-13-TEX-B-1D431" d="M227 0Q212 3 121 3Q40 3 28 0H21V62H117L245 213L109 382H26V444H34Q49 441 143 441Q247 441 265 444H274V382H246L281 339Q315 297 316 297Q320 297 354 341L389 382H352V444H360Q375 441 466 441Q547 441 559 444H566V382H471L355 246L504 63L545 62H586V0H578Q563 3 469 3Q365 3 347 0H338V62H366Q366 63 326 112T285 163L198 63L217 62H235V0H227Z"></path><path id="MJX-13-TEX-N-31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path><path id="MJX-13-TEX-N-32" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z"></path><path id="MJX-13-TEX-N-22EE" d="M78 30Q78 54 95 72T138 90Q162 90 180 74T199 31Q199 6 182 -12T139 -30T96 -13T78 30ZM78 440Q78 464 95 482T138 500Q162 500 180 484T199 441Q199 416 182 398T139 380T96 397T78 440ZM78 840Q78 864 95 882T138 900Q162 900 180 884T199 841Q199 816 182 798T139 780T96 797T78 840Z"></path><path id="MJX-13-TEX-I-1D45B" d="M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z"></path><path id="MJX-13-TEX-S4-23A4" d="M0 1070V1154H347V-645H263V1070H0Z"></path><path id="MJX-13-TEX-S4-23A6" d="M263 -560V1155H347V-644H0V-560H263Z"></path><path id="MJX-13-TEX-S4-23A5" d="M263 0V602H347V0H263Z"></path><path id="MJX-13-TEX-I-1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z"></path><path id="MJX-13-TEX-N-2C" d="M78 35T78 60T94 103T137 121Q165 121 187 96T210 8Q210 -27 201 -60T180 -117T154 -158T130 -185T117 -194Q113 -194 104 -185T95 -172Q95 -168 106 -156T131 -126T157 -76T173 -3V9L172 8Q170 7 167 6T161 3T152 1T140 0Q113 0 96 17Z"></path><path id="MJX-13-TEX-N-22EF" d="M78 250Q78 274 95 292T138 310Q162 310 180 294T199 251Q199 226 182 208T139 190T96 207T78 250ZM525 250Q525 274 542 292T585 310Q609 310 627 294T646 251Q646 226 629 208T586 190T543 207T525 250ZM972 250Q972 274 989 292T1032 310Q1056 310 1074 294T1093 251Q1093 226 1076 208T1033 190T990 207T972 250Z"></path><path id="MJX-13-TEX-I-1D451" d="M366 683Q367 683 438 688T511 694Q523 694 523 686Q523 679 450 384T375 83T374 68Q374 26 402 26Q411 27 422 35Q443 55 463 131Q469 151 473 152Q475 153 483 153H487H491Q506 153 506 145Q506 140 503 129Q490 79 473 48T445 8T417 -8Q409 -10 393 -10Q359 -10 336 5T306 36L300 51Q299 52 296 50Q294 48 292 46Q233 -10 172 -10Q117 -10 75 30T33 157Q33 205 53 255T101 341Q148 398 195 420T280 442Q336 442 364 400Q369 394 369 396Q370 400 396 505T424 616Q424 629 417 632T378 637H357Q351 643 351 645T353 664Q358 683 366 683ZM352 326Q329 405 277 405Q242 405 210 374T160 293Q131 214 119 129Q119 126 119 118T118 106Q118 61 136 44T179 26Q233 26 290 98L298 109L352 326Z"></path><path id="MJX-13-TEX-B-1D432" d="M84 -102Q84 -110 87 -119T102 -138T133 -149Q148 -148 162 -143T186 -131T206 -114T222 -95T234 -76T243 -59T249 -45T252 -37L269 0L96 382H26V444H34Q49 441 146 441Q252 441 270 444H279V382H255Q232 382 232 380L337 151L442 382H394V444H401Q413 441 495 441Q568 441 574 444H580V382H510L406 152Q298 -84 297 -87Q269 -139 225 -169T131 -200Q85 -200 54 -172T23 -100Q23 -64 44 -50T87 -35Q111 -35 130 -50T152 -92V-100H84V-102Z"></path><path id="MJX-13-TEX-N-28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z"></path><path id="MJX-13-TEX-I-1D466" d="M21 287Q21 301 36 335T84 406T158 442Q199 442 224 419T250 355Q248 336 247 334Q247 331 231 288T198 191T182 105Q182 62 196 45T238 27Q261 27 281 38T312 61T339 94Q339 95 344 114T358 173T377 247Q415 397 419 404Q432 431 462 431Q475 431 483 424T494 412T496 403Q496 390 447 193T391 -23Q363 -106 294 -155T156 -205Q111 -205 77 -183T43 -117Q43 -95 50 -80T69 -58T89 -48T106 -45Q150 -45 150 -87Q150 -107 138 -122T115 -142T102 -147L99 -148Q101 -153 118 -160T152 -167H160Q177 -167 186 -165Q219 -156 247 -127T290 -65T313 -9T321 21L315 17Q309 13 296 6T270 -6Q250 -11 231 -11Q185 -11 150 11T104 82Q103 89 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z"></path><path id="MJX-13-TEX-N-2026" d="M78 60Q78 84 95 102T138 120Q162 120 180 104T199 61Q199 36 182 18T139 0T96 17T78 60ZM525 60Q525 84 542 102T585 120Q609 120 627 104T646 61Q646 36 629 18T586 0T543 17T525 60ZM972 60Q972 84 989 102T1032 120Q1056 120 1074 104T1093 61Q1093 36 1076 18T1033 0T990 17T972 60Z"></path><path id="MJX-13-TEX-N-29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z"></path></defs><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><g data-mml-node="math"><g data-mml-node="mtable"><g data-mml-node="mtr"><g data-mml-node="mtd"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D44B"></use></g><g data-mml-node="mo" transform="translate(1129.8, 0)"><use xlink:href="#MJX-13-TEX-N-3D"></use></g><g data-mml-node="mrow" transform="translate(2185.6, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-S4-23A1" transform="translate(0, 1971)"></use><use xlink:href="#MJX-13-TEX-S4-23A3" transform="translate(0, -1981)"></use><svg width="667" height="2352" y="-926" x="0" viewBox="0 588 667 2352"><use xlink:href="#MJX-13-TEX-S4-23A2" transform="scale(1, 5.86)"></use></svg></g><g data-mml-node="mtable" transform="translate(667, 0)"><g data-mml-node="mtr" transform="translate(0, 2375)"><g data-mml-node="mtd" transform="translate(35.4, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-2212"></use></g><g data-mml-node="msub" transform="translate(778, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-B-1D431"></use></g></g><g data-mml-node="mn" transform="translate(607, -150) scale(0.707)"><use xlink:href="#MJX-13-TEX-N-31"></use></g></g><g data-mml-node="mo" transform="translate(1788.6, 0)"><use xlink:href="#MJX-13-TEX-N-2212"></use></g></g></g><g data-mml-node="mtr" transform="translate(0, 975)"><g data-mml-node="mtd" transform="translate(35.4, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-2212"></use></g><g data-mml-node="msub" transform="translate(778, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-B-1D431"></use></g></g><g data-mml-node="mn" transform="translate(607, -150) scale(0.707)"><use xlink:href="#MJX-13-TEX-N-32"></use></g></g><g data-mml-node="mo" transform="translate(1788.6, 0)"><use xlink:href="#MJX-13-TEX-N-2212"></use></g></g></g><g data-mml-node="mtr" transform="translate(0, -975)"><g data-mml-node="mtd" transform="translate(1179.6, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-22EE"></use></g></g></g></g><g data-mml-node="mtr" transform="translate(0, -2375)"><g data-mml-node="mtd"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-2212"></use></g><g data-mml-node="msub" transform="translate(778, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-B-1D431"></use></g></g><g data-mml-node="mi" transform="translate(607, -150) scale(0.707)"><use xlink:href="#MJX-13-TEX-I-1D45B"></use></g></g><g data-mml-node="mo" transform="translate(1859.3, 0)"><use xlink:href="#MJX-13-TEX-N-2212"></use></g></g></g></g><g data-mml-node="mo" transform="translate(3304.3, 0)"><use xlink:href="#MJX-13-TEX-S4-23A4" transform="translate(0, 1971)"></use><use xlink:href="#MJX-13-TEX-S4-23A6" transform="translate(0, -1981)"></use><svg width="667" height="2352" y="-926" x="0" viewBox="0 588 667 2352"><use xlink:href="#MJX-13-TEX-S4-23A5" transform="scale(1, 5.86)"></use></svg></g></g><g data-mml-node="mo" transform="translate(6434.6, 0)"><use xlink:href="#MJX-13-TEX-N-3D"></use></g><g data-mml-node="mrow" transform="translate(7490.4, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-S4-23A1" transform="translate(0, 2026.8)"></use><use xlink:href="#MJX-13-TEX-S4-23A3" transform="translate(0, -2036.8)"></use><svg width="667" height="2463.5" y="-981.8" x="0" viewBox="0 615.9 667 2463.5"><use xlink:href="#MJX-13-TEX-S4-23A2" transform="scale(1, 6.138)"></use></svg></g><g data-mml-node="mtable" transform="translate(667, 0)"><g data-mml-node="mtr" transform="translate(0, 2430.8)"><g data-mml-node="mtd" transform="translate(35.4, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-13-TEX-N-31"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(2631.7, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-13-TEX-N-32"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(5192.8, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(7400.1, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mi" transform="translate(778, 0)"><use xlink:href="#MJX-13-TEX-I-1D451"></use></g></g></g></g></g><g data-mml-node="mtr" transform="translate(0, 993.6)"><g data-mml-node="mtd" transform="translate(35.4, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-32"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-13-TEX-N-31"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(2631.7, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-32"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-13-TEX-N-32"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(5192.8, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(7400.1, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-32"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mi" transform="translate(778, 0)"><use xlink:href="#MJX-13-TEX-I-1D451"></use></g></g></g></g></g><g data-mml-node="mtr" transform="translate(0, -993.6)"><g data-mml-node="mtd" transform="translate(659.2, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-22EE"></use></g></g></g><g data-mml-node="mtd" transform="translate(3394.6, 0)"></g><g data-mml-node="mtd" transform="translate(5778.8, 0)"></g><g data-mml-node="mtd" transform="translate(8031.1, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-22EE"></use></g></g></g></g><g data-mml-node="mtr" transform="translate(0, -2393.6)"><g data-mml-node="mtd"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D45B"></use></g><g data-mml-node="mo" transform="translate(600, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(878, 0)"><use xlink:href="#MJX-13-TEX-N-31"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(2596.4, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D45B"></use></g><g data-mml-node="mo" transform="translate(600, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(878, 0)"><use xlink:href="#MJX-13-TEX-N-32"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(5192.8, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-13-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(7364.8, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D45B"></use></g><g data-mml-node="mo" transform="translate(600, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mi" transform="translate(878, 0)"><use xlink:href="#MJX-13-TEX-I-1D451"></use></g></g></g></g></g></g><g data-mml-node="mo" transform="translate(9642.3, 0)"><use xlink:href="#MJX-13-TEX-S4-23A4" transform="translate(0, 2026.8)"></use><use xlink:href="#MJX-13-TEX-S4-23A6" transform="translate(0, -2036.8)"></use><svg width="667" height="2463.5" y="-981.8" x="0" viewBox="0 615.9 667 2463.5"><use xlink:href="#MJX-13-TEX-S4-23A5" transform="scale(1, 6.138)"></use></svg></g></g><g data-mml-node="mstyle" transform="translate(17799.7, 0)"><g data-mml-node="mspace"></g></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(18799.7, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-B-1D432"></use></g></g><g data-mml-node="mo" transform="translate(19684.5, 0)"><use xlink:href="#MJX-13-TEX-N-3D"></use></g><g data-mml-node="mo" transform="translate(20740.3, 0)"><use xlink:href="#MJX-13-TEX-N-28"></use></g><g data-mml-node="msub" transform="translate(21129.3, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D466"></use></g><g data-mml-node="TeXAtom" transform="translate(490, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-31"></use></g></g></g><g data-mml-node="mo" transform="translate(22022.8, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="msub" transform="translate(22467.5, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D466"></use></g><g data-mml-node="TeXAtom" transform="translate(490, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-13-TEX-N-32"></use></g></g></g><g data-mml-node="mo" transform="translate(23361, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="mo" transform="translate(23805.7, 0)"><use xlink:href="#MJX-13-TEX-N-2026"></use></g><g data-mml-node="mo" transform="translate(25144.4, 0)"><use xlink:href="#MJX-13-TEX-N-2C"></use></g><g data-mml-node="msub" transform="translate(25589, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D466"></use></g><g data-mml-node="TeXAtom" transform="translate(490, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-13-TEX-I-1D45B"></use></g></g></g><g data-mml-node="mo" transform="translate(26553.3, 0)"><use xlink:href="#MJX-13-TEX-N-29"></use></g></g></g></g></g></g></svg>

其中 $x_{i,j}$ 表示第 $i$ 个样本 $\mathbf {x}_i$ 的第 $j$ 维特征，$y_i$ 为样本 $\mathbf {x}_i$ 对应的标签。

- **不同的人会对数学符号的使用有着不同的约定，一定需要搞清楚这些符号的定义，或者在交流时采用比较通用的定义方式**
- 为了简洁美观，我们这里用粗体表示向量 $\mathbf {x}$; 而在手写时粗细体并不容易辨别，所以板书时常用上标箭头来表示向量 $\vec {x}$.

```python
X, y = boston.data, boston.target
num, dim = X.shape

print(X[0], y[0]) # 取出第一个样本，查看对应的特征向量和标签值
```

输出结果：

```
[6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01
 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00] 24.0
```

### 数据集的几种类型

对于常见的机器学习任务，用作基准（Benchmark）的数据集通常会被分为训练集（Training dataset）和测试集（Test dataset）两部分；

- 我们利用训练集的数据，通过优化损失函数的形式将模型训练好，在测试集上对模型进行测试，根据选用的评估指标（Metrics）评价模型性能
- 测试集的数据不能参与模型训练，因为我们希望测试集数据代表着将来会被用于预测的真实数据，它们现在是“不可见的”，当作标签未知

训练模型时为了调整和选取合适的超参数，通常还会在训练集的基础上划分出验证集（Validation dataset）或者说开发集（Develop dataset）

- 在一些机器学习竞赛中，比赛方会将测试集中一部分拿出来做为 Public Leaderboard 评分和排名，剩下的部分作为 Private Leaderboard 的评分和排名。
- 选手可以根据 Public Leaderboard 上的评估结果及时地对算法和超参数进行调整优化，但最终的排名将以 Private Leaderboard 为准
- 对应地，可以将 Public Leaderboard 所使用的数据集理解为验证集，将 Private Leaderboard 所使用的数据集理解为测试集
- 为了避免在短时间内引入过多密集的新知识，我们目前将不会进行验证集和开发集的代码实践和讨论

数据集该如何划分是一个值得讨论与研究的话题，它是数据预处理的一个环节，目前我们只需要对它有一个基本概念，通过不断实践来加深理解。

对于波士顿房价数据集，我们可以将 506 张样本划分为 500 张训练样本，6 张测试样本：

```python
mask = 500  # 这个值是随意设计的，不用想太多背后的原因

train_dataset = X[:mask]
test_dataset = X[mask:num]
train_label = y[:mask]
test_label = y[mask:num]

print(len(train_dataset), len(train_label))
print(len(test_dataset), len(test_label))
```

输出结果：

```
500 500
6 6
```

## 模型定义、训练和测试

对于单个的样本 $\mathbf {x}$, 想要预测输出 $y$, 即尝试找到映射关系 $f: \mathbf {x} \in \mathbb {R}^{d} \mapsto y$, 同样可以建立线性模型：

$$
\begin{aligned}
y &= f(\mathbf {x}) = \mathbf {w} \cdot \mathbf {x} + b \\
& = (w_{1}, w_{2}, \ldots, w_{13}) \cdot (x_{1}, x_{2}, \ldots, x_{13}) + b\\
& = w_{1} x_{1} + w_{2} x_{2} + \ldots + w_{13} x_{13} + b
\end{aligned}
$$

**注意在表示不同的乘积（Product）时，不仅使用的符号有所不同，数学概念和编程代码之间又有所区别：** 

- 上个教程中用的星乘 $*$ 符号指的是元素间相乘（有时用 $\odot$ 表示），也适用于标量乘法，编程对应于 `np.multiply()` 方法
- 对于点乘我们用 $\cdot$ 符号表示，编程对应于 `np.dot()` 方法，在作用于两个向量时它等同于内积 `np.inner()` 方法
  - 根据传入参数的形状，`np.dot()` 的行为也可以等同于矩阵乘法，对应于 `numpy.matmul()` 方法或 `@` 运算符
  - 严谨地说，点积或者数量积只是内积的一种特例，**但我们在编程时，应该习惯先查询文档中的说明，并进行简单验证**

比如样本 $\mathbf {x}$ 和参数 $\mathbf {w}$ 都是 13 维的向量，二者的点积结果是一个标量：

```python
w = X[0]
x = np.random.random((13,))
print(x.shape, w.shape) # 一维 ndarray，即为向量

p = np.dot(x, w)        # 此时 dot() 逻辑是进行向量点乘
print(p.shape)          # 零维 ndarray，即为标量
```

输出结果：

```
(13,) (13,)
()
```

我们之前已经实现过类似的训练过程，现在一起来看下面这份新实现的代码：

```python
epochs = 5
lr = 1e-6

w = np.zeros((13,))
b = np.zeros(())

data = train_dataset
label = train_label
n = len(data)

def linear_model(x):
    return np.dot(x, w) + b   # 已经不再是 x * w 

for epoch in range(epochs):
    
    loss = 0
    
    sum_grad_w = 0
    sum_grad_b = 0
    
    for i in range(n):
        
        pred = linear_model(data[i])
        loss += (pred - label[i]) ** 2
        
        sum_grad_w += 2 * (pred - label[i]) * data[i]
        sum_grad_b += 2 * (pred - label[i])
    
    grad_w = sum_grad_w / n
    grad_b = sum_grad_b / n
    
    w = w - lr * grad_w
    b = b - lr * grad_b
    
    loss = loss / n
    print("epoch = {}, loss = {:.3f}".format(epoch, loss))
```

输出结果：

```
epoch = 0, loss = 594.442
epoch = 1, loss = 197.173
epoch = 2, loss = 138.077
epoch = 3, loss = 126.176
epoch = 4, loss = 121.148
```

现在，我们可以使用训练得到的 `w` 和 `b` 在测试数据上进行评估，这里我们可以选择使用平均绝对误差（Mean Absolute Error, MAE）作为评估指标：

$$
\ell(y_{pred}, y_{real})= \frac{1}{n }\sum_{i=1}^{n}\left | \hat{y}\_{i}-{y}_{i}\right |
$$

```python
data = test_dataset
label = test_label
n = len(data)

loss = 0

for i in range(n):
    pred = linear_model(data[i])
    loss += np.abs(pred - label[i])
    print("Pred = {:.3f}, Real = {:.3f}".format(pred, label[i]))

loss = loss / n

print("Average error = {:.3f}".format(loss))
```

输出结果：

```
Pred = 21.814, Real = 16.800
Pred = 18.942, Real = 22.400
Pred = 19.130, Real = 20.600
Pred = 19.191, Real = 23.900
Pred = 19.075, Real = 22.000
Pred = 19.148, Real = 11.900
Average error = 4.137
```

### 不同的梯度下降形式

我们可以发现：上面的代码中，我们每训练一个完整的 `epoch`, 将根据所有样本的平均梯度进行一次参数更新。

这种通过对所有的样本的计算来求解梯度的方法叫做**批梯度下降法(Batch Gradient Descent)**.

当碰到样本容量特别大的情况时，可能会导致无法一次性将所有数据给读入内存，遇到内存用尽（Out of memory，OOM）的情况。

这时你可能会想：“其实我们完全可以在每个样本经过前向传播计算损失、反向传播计算得到梯度后时，就立即对参数进行更新呀！”

Bingo~ 这种思路叫做**随机梯度下降（Stochastic Gradient Descent，常缩写成 SGD)**，动手改写后，整体代码实现如下：


```python
epochs = 5
lr = 1e-6

w = np.zeros((13,))
b = np.zeros(())

data = train_dataset
label = train_label
n = len(data)

def linear_model(x):
    return np.dot(x, w) + b

for epoch in range(epochs):
    
    loss = 0
    
    for i in range(n):
        
        pred = linear_model(data[i])
        loss += (pred - label[i]) ** 2
        
        w -= lr * 2 * (pred - label[i]) * data[i]
        b -= lr * (pred - label[i])
    
    loss = loss / n
    
    print("epoch = {}, loss = {:.3f}".format(epoch, loss))
```

输出结果：

```
epoch = 0, loss = 45.178
epoch = 1, loss = 42.856
epoch = 2, loss = 42.358
epoch = 3, loss = 41.948
epoch = 4, loss = 41.595
```

可以看到，在同样的训练周期内，使用随机梯度下降得到的训练损失更低，即损失收敛得更快。这是因为参数的实际更新次数要多得多。

接下来我们用随机梯度下降得到的参数 `w` 和 `b` 进行测试：

```python
data = test_dataset
label = test_label
n = len(data)

loss = 0

for i in range(n):
    pred = linear_model(data[i])
    loss += np.abs(pred - label[i])
    print("Pred = {:.3f}, Real = {:.3f}".format(pred, label[i]))

loss = loss / n

print("Average error = {:.3f}".format(loss))
```

输出结果：

```
Pred = 18.598, Real = 16.800
Pred = 16.334, Real = 22.400
Pred = 16.458, Real = 20.600
Pred = 16.556, Real = 23.900
Pred = 16.423, Real = 22.000
Pred = 16.492, Real = 11.900
Average error = 4.920
```

可以看到，虽然我们在训练集上的损失远低于使用批梯度下降的损失，但测试时得到的损失反而略高于批梯度下降的测试结果。为什么会这样？

**抛开数据、模型、损失函数和评估指标本身的因素不谈，背后的原因可能是：**

- 使用随机梯度下降时，考虑到噪声数据的存在，并不一定参数每次更新都是朝着模型整体最优化的方向；
- 若样本噪声较多，很容易陷入局部最优解而收敛到不理想的状态；
- 如果更新次数过多，还容易出现在训练数据过拟合的情况，导致泛化能力变差。

既然两种梯度下降策略各有千秋，因此有一种折衷的方式，即采用**小批量梯度下降法（Mini-Batch Gradient Descent）**：

- 我们设定一个 `batch_size` 值，将数据划分为多个 `batch`，每个 `batch` 的数据采取批梯度下降策略来更新参数；
- 设置合适的 `batch_size` 值，既可以避免出现内存爆掉的情况，也使得损失可能更加平滑地收敛；
- 不难发现 `batch_size` 是一个超参数；当 `batch_size=1` 时，小批量梯度下降其实就等同于随机梯度下降

注意：天元 MegEngine 的优化器 `Optimizer` 中实现的 `SGD` 优化策略，实际上就是对小批量梯度下降法逻辑的实现。

这种折衷方案的效果比“真”随机梯度下降要好得多，**因为可以利用向量化加速批数据的运算，而不是分别计算每个样本。**

在这里我们先卖个关子，把向量化的介绍放在更后面，因为我们的当务之急是：获取小批量数据（Mini-batch data）.

### 采样器（Sampler）

想要从完整的数据集中获得小批量的数据，则需要对已有数据进行采样。

在 MegEngine 的 `data` 模块中，提供了多种采样器 `Sampler`：

- 对于训练数据，通常使用顺序采样器 `SequentialSampler`, 我们即将用到；
- 对于测试数据，通常使用 `RandomSampler` 进行随机采样，在本教程的测试部分就能见到；
- 对于 `dataset` 的样本容量不是 `batch_size` 整数倍的情况，采样器也能进行很好的处理

```python
from megengine.data import SequentialSampler

sequential_sampler = SequentialSampler(dataset=train_dataset, batch_size=100)

# SequentialSampler 每次返回的是顺序索引，而不是划分后的数据本身
for indices in sequential_sampler:
    print(indices[0:5], "...", indices[95:100])
```

```
[0, 1, 2, 3, 4] ... [95, 96, 97, 98, 99]
[100, 101, 102, 103, 104] ... [195, 196, 197, 198, 199]
[200, 201, 202, 203, 204] ... [295, 296, 297, 298, 299]
[300, 301, 302, 303, 304] ... [395, 396, 397, 398, 399]
[400, 401, 402, 403, 404] ... [495, 496, 497, 498, 499]
```

但需要注意到，采样器 `Sampler` 返回的是索引值，要得到划分后的批数据，还需要结合使用 MegEngine 中的 `Dataloader` 模块。

同时也要注意，如果想要使用 `Dataloader`, 需要先将原始数据集变成 MegEngine 支持的 `Dataset` 对象。

## 利用 Dataset 封装一个数据集

`Dataset` 是 MegEngine 中表示数据集最原始的的抽象类，可被其它类继承（如 `StreamDataset`），可阅读 `Dataset` 模块的文档了解更多的细节。

通常我们自定义的数据集类应该继承 `Dataset` 类，并重写下列方法：

- `__init__()` ：一般在其中实现读取数据源文件的功能。也可以添加任何其它的必要功能；
- `__getitem__()` ：通过索引操作来获取数据集中某一个样本，使得可以通过 for 循环来遍历整个数据集；
- `__len__()` ：返回数据集大小

```python
from typing import Tuple
from megengine.data.dataset import Dataset

class BostonTrainDataset(Dataset):
    def __init__(self):
        self.data = train_dataset
        self.label = train_label
    
    def __getitem__(self, index: int) -> Tuple:
        return self.data[index], self.label[index]

    def __len__(self) -> int:
        return len(self.data)

boston_train_dataset = BostonTrainDataset()
print(len(boston_train_dataset))
```

其实，对于这种单个或多个 NumPy 数组构成的数据，在 MegEngine 中也可以使用 `ArrayDataset` 对它进行初始化，它将自动完成以上方法的重写：

```python
from megengine.data.dataset import ArrayDataset

boston_train_dataset = ArrayDataset(train_dataset, train_label)
print(len(boston_train_dataset))
```

输出结果：

```
500
```

### 数据载入器（Dataloader）

接下来便可以通过 `Dataloader` 来生成批数据，每个元素由对应的 `batch_data` 和 `batch_label` 组成：

```python
from megengine.data import DataLoader

sequential_sampler = SequentialSampler(dataset=boston_train_dataset, batch_size=100)
train_dataloader = DataLoader(dataset=boston_train_dataset, sampler=sequential_sampler)

for batch_data, batch_label in train_dataloader:
    print(batch_data.shape, batch_label.shape, len(train_dataloader))
    break
```

输出结果：

```
(100, 13) (100,) 5
```

接下来我们一起来看看，使用批数据为什么能够加速整体的运算效率。

## 通过向量化加速运算

在 NumPy 内部，向量化运算的速度是优于 `for` 循环的，我们很容易验证这一点：

```python
import time

n = 1000000
a = np.random.rand(n)
b = np.random.rand(n)
c = np.zeros(n)

time_start = time.time()
for i in range(n):
    c[i] = a[i] * b[i]
time_end = time.time()
print('For loop version:', str(1000 * (time_end - time_start)), 'ms')

time_start = time.time()
c = np.dot(a, b)
time_end = time.time()
print('Vectorized version:', str(1000 * (time_end - time_start)), 'ms')
```

输出结果：

```
For loop version: 354.91228103637695 ms
Vectorized version: 1.2269020080566406 ms
```

重新阅读模型训练的代码，不难发现，每个 `epoch` 内部存在着 `for` 循环，根据模型定义的 $y_i = \mathbf {w} \cdot \mathbf {x_i} + b$ 进行了 $n$ 次计算。

我们在前面已经将数据集表示成了形状为 $(n, d)$ 的数据矩阵 $X$, 将标签表示成了 $y$ 向量，这启发我们一次性完成所有样本的前向计算过程：

$$
(y_{1}, y_{2}, \ldots, y_{n}) = 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
\vdots  &         &        & \vdots \\
x_{n,1} & x_{n,2} & \cdots & x_{n,d} 
\end{bmatrix}
\cdot (w_{1}, w_{2}, \ldots, w_{d}) + 
b
$$

一种比较容易理解的形式是将其看成是矩阵运算 $Y=XW+B$：

$$
\begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{n}
\end{bmatrix} = 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
\vdots  &         &        & \vdots \\
x_{n,1} & x_{n,2} & \cdots & x_{n,d} 
\end{bmatrix}
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_d
\end{bmatrix} + 
\begin{bmatrix}
b \\
b \\
\vdots \\
b
\end{bmatrix}
$$

- 形状为 $(n,d)$ 的矩阵 $X$ 和维度为 $(d,)$ 的向量 $w$ 进行点乘，此时不妨理解成 $w$ 变成了一个形状为 $(d, 1)$ 的矩阵 $W$
- 两个矩阵进行乘法运算，此时的 `np.dot(X, w)` 等效于 `np.matmul(X, W)`, 底层效率比 `for` 循环快许多，得到形状为 $(n, 1)$ 的中间矩阵 $P$
- 中间矩阵 $P$ 和标量 $b$ 相加，此时标量 $b$ 广播成 $(n, 1)$ 的矩阵 $B$ 进行矩阵加法，得到形状为 $(n,1)$ 的标签矩阵 $Y$
- 显然，这样的处理逻辑还需要将标签矩阵 $Y$ 去掉冗余的那一维，变成形状为 $(n, )$ 的标签向量 $y$

**矩阵/张量运算的思路在深度学习中十分常见，在后续会被反复提及。** NumPy 的官方文档中写着此时 `np.dot()` 的真实逻辑：

- If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
- 如果 `a` 是 N 维度数组且 `b` 是 1 维数组，将对 `a` 和 `b` 最后一轴上对元素进行乘积并求和。

这样的计算逻辑也可以由爱因斯坦求和约定 `np.einsum('ij,j->i', X, w)` 来实现，感兴趣的读者可查阅 `np.einsum()` 文档。

我们可以设计一套简单的测试样例，来看一下 `np.dot()`, `np.einsum()` 和 `np.matmul()` 相较于 `for` 循环是否能得到同样的结果：

```python
X = np.random.random((3, 2))
w = np.random.random((2, ))            # 在 NumPy 中向量有三种表现形式，这里的 w 是一维数组（1-D ndarray），即向量
W = w.reshape((2, 1))                  # 这里的 W 已经是二维数组（2-D ndarray）的“列”向量，即矩阵的特殊形式之一；“行”向量同理
b = np.random.random(())
y = np.empty((3, ))

for i in range(3):
    y[i] = np.dot(X[i], w) + b         # 此时 dot() 等同于向量内积，官方推荐 np.inner(a, b) 写法
    
print("For loop version:", y)
print("np.dot() version:", np.dot(X, w) + b)                
print("np.einsum() version:", np.einsum('ij,j->i', X, w) + b)
print("np.matmul() version:", np.squeeze(np.matmul(X, W) + b))   # np.squeeze() 可以去掉冗余的那一维
```

输出结果：

```
For loop version: [0.93044708 0.86239961 1.1775819 ]
np.dot() version: [0.93044708 0.86239961 1.1775819 ]
np.einsum() version: [0.93044708 0.86239961 1.1775819 ]
np.matmul() version: [0.93044708 0.86239961 1.1775819 ]
```

接下来加大数据规模，测试向量化是否起到了加速效果：

```python
n = 50000
d = 1000

X = np.random.random((n, d))
w = np.random.random((d))
W = w.reshape((d, 1))
b = np.random.random(())
B = b * np.ones((n, 1))    # 提前生成 B 矩阵，避免将广播操作的开销计算在内
y = np.empty((n, ))


time_start = time.time()
for i in range(n):
    y[i] = np.dot(X[i], w) + b
time_end = time.time()
print("For loop version:", str(1000 * (time_end - time_start)), 'ms')

time_start = time.time()
y = np.dot(X, w) + b
time_end = time.time()
print("np.dot() version:", str(1000 * (time_end - time_start)), 'ms')

time_start = time.time()
y = np.einsum('ij,j->i', X, w) + b
time_end = time.time()
print("np.einsum() version:", str(1000 * (time_end - time_start)), 'ms')

time_start = time.time()
Y = np.matmul(X, W) + B
time_end = time.time()
print("np.matmul() version:", str(1000 * (time_end - time_start)), 'ms')
```

输出结果：

```
For loop version: 139.1618251800537 ms
np.dot() version: 9.843826293945312 ms
np.einsum() version: 36.226511001586914 ms
np.matmul() version: 5.5789947509765625 ms
```

可以发现 `dot()` 和 `matmul()` 的向量化实现都有明显的加速效果，`einsum()` 虽然是全能型选手，但也以更多的开销作为了代价，一般不做考虑。

### NumPy 实现

我们先尝试使用 NumPy 的 `dot()` 实现一下向量化的批梯度下降的代码：

```python
epochs = 5
lr = 1e-6

w = np.zeros((13,))
b = np.zeros(())

data = train_dataset
label = train_label

def linear_model(x):
    return np.dot(x, w) + b

for epoch in range(epochs):

    pred = linear_model(data)
    loss = np.sum((pred - label) ** 2)
    
    # 对应地，反向传播计算梯度的代码也需要改成向量化形式，这里不解释矩阵求导的推导过程
    w -= lr * 2 * np.dot(data.T, (pred - label)) / len(data)
    b -= lr * 2 * np.mean(pred - label)

    print("epoch = {}, loss = {:.3f}".format(epoch, loss / len(data)))
```

输出结果：

```
epoch = 0, loss = 594.442
epoch = 1, loss = 197.173
epoch = 2, loss = 138.077
epoch = 3, loss = 126.176
epoch = 4, loss = 121.148
```

上面的 `loss` 收敛状况与我们最开始实现的 `sum_grad` 的数值一致，可以自行测试一下 `for` 循环写法和向量化写法的用时差异。

同时我们也可以发现，当前向传播使用批数据变成矩阵计算带来极大加速的同时，也为我们的反向传播梯度计算提出了更高的要求：

- 尽管存在着链式法则，但是对于不熟悉矩阵求导的人来说，还是很难理解一些梯度公式推导的过程，比如为什么出现了转置 `data.T` 
- 当前向传播的计算变得更加复杂，由框架实现自动求导 `autograde` 机制就显得更加必要和方便

### MegEngine 实现

最后，让我们利用上小批量（Mini-batch）的数据把完整的训练流程代码在 MegEngine 中实现：

- 为了后续教程理解的连贯性，我们做一些改变，使用 `F.matmul()` 来代替 `np.dot()`，前面已经证明了这种情况下的计算结果值是等价的，但形状不同
- 对应地，我们的 `w` 将由向量变为矩阵，`b` 在进行加法操作时将自动进行广播变成矩阵，输出 `pred` 也是一个矩阵，即 2 维 Tensor
- 在计算单个 `batch` 的 `loss` 时，内部通过 `sum()` 计算已经去掉了冗余的维度，最终得到的 `loss` 是一个 0 维 Tensor

```python
import numpy as np
from sklearn.datasets import load_boston
import megengine as mge
import megengine.functional as F
from megengine.data.dataset import ArrayDataset
from megengine.data import SequentialSampler, RandomSampler, DataLoader
from megengine.autodiff import GradManager
import megengine.optimizer as optim

# 设置超参数
bs = 100
lr = 1e-6
epochs = 5
mask = 500

# 读取原始数据集
boston = load_boston()
X, y = boston.data, boston.target
total_num, num_features = boston.data.shape

# 训练数据加载与预处理
boston_train_dataset = ArrayDataset(X[:mask], y[:mask])
train_sampler = SequentialSampler(dataset=boston_train_dataset, batch_size=bs)
train_dataloader = DataLoader(dataset=boston_train_dataset, sampler=train_sampler)

# 初始化参数
W = mge.Parameter(np.zeros((num_features, 1)))
b = mge.Parameter(np.zeros(()))

# 定义模型
def linear_model(X):
    return F.matmul(X, W) + b

# 定义求导器和优化器
gm = GradManager().attach([W, b])
optimizer = optim.SGD([W, b], lr=lr)

# 模型训练
for epoch in range(epochs):
    total_loss = 0
    for batch_data, batch_label in train_dataloader:
        batch_data = mge.tensor(batch_data)
        batch_label = mge.tensor(batch_label)
        with gm:
            pred = linear_model(batch_data)
            loss = F.loss.square_loss(pred, batch_label)
            gm.backward(loss)
        optimizer.step().clear_grad()
        total_loss +=  loss.item()
    print("epoch = {}, loss = {:.3f}".format(epoch, total_loss / len(train_dataloader)))
```

输出结果：

```
epoch = 0, loss = 261.126
epoch = 1, loss = 135.167
epoch = 2, loss = 122.774
epoch = 3, loss = 115.320
epoch = 4, loss = 110.772
```

对我们训练好的模型进行测试：

```python
boston_test_dataset = ArrayDataset(X[mask:num], y[mask:num])
test_sampler = RandomSampler(dataset=boston_test_dataset, batch_size=1)  # 测试时通常随机采样
test_dataloader = DataLoader(dataset=boston_test_dataset, sampler=test_sampler)

loss = 0
for batch_data, batch_label in test_dataloader:
    batch_data = mge.tensor(batch_data)
    batch_label = mge.tensor(batch_label)
    pred = linear_model(batch_data)
    print("Pred = {:.3f}, Real = {:.3f}".format(pred.item(), batch_label.item()))
    loss += np.abs(pred.item() - batch_label.item())
print("Average error = {:.3f}".format(loss / len(test_dataloader)))
```

输出结果：

```
Pred = 19.251, Real = 23.900
Pred = 19.173, Real = 20.600
Pred = 19.091, Real = 22.000
Pred = 19.195, Real = 11.900
Pred = 19.733, Real = 16.800
Pred = 18.916, Real = 22.400
Average error = 3.783
```

可以发现，使用小批量梯度下降策略更新参数，最终在测试时得到的平均绝对值损失比单独采用批梯度下降和随机梯度下降策略时还要好一些。

## 总结回顾

有些时候，添加一些额外的条件，看似简单的问题就变得更加复杂有趣起来了，启发我们进行更多的思考和探索。

我们需要及时总结学习过的知识，并在之后的时间内不断通过实践来强化记忆，这次的教程中出现了以下新的概念：

- 数据集（Dataset）：想要将现实中的数据集变成能够被 MegEngine 使用的格式，必须将其处理成 `MapDataset` 或者 `StreamDateset` 类
  - 继承基类的同时，还需要实现 `__init__()`, `__getitem__` 和 `__len__()` 三种内置方法
  - 我们对事物的抽象——特征（Feature）有了一定的认识，并接触了一些数学形式的表达如数据矩阵 $X$ 和标签向量 $y$ 等
  - 我们接触到了训练集（Trainining dataset）、测试集（Test dataset）和验证集（Validation dataset）/ 开发集（Develop dataset）的概念
  - 除了训练时我们会设计一个损失函数外，在测试模型性能时，我们也需要设计一个评估指标（Metric），不同任务的指标不尽相同
- 采样器（Sampler）：可以帮助我们自动地获得一个采样序列，常见的有顺序采样器 `SequentialSampler` 和随机采样器 `RandomSampler`
- 数据载入器（Dataloader）：根据 `Dataset` 和 `Sampler` 来获得对应的小批量数据（Mini-batch data）
  - 我们认识了一个新的超参数 `batch_size`，并且对不同形式的梯度下降各自的优缺点有了一定的认识
  - 我们理解了批数据向量化计算带来的好处，能够大大加快计算效率；同时为了避免推导向量化的梯度，实现自动求导机制是相当必要的

我们要养成及时查阅文档的好习惯，比如 NumPy 的 `np.dot()` 在接受的 ndarray 形状不同时，运算逻辑也将发生变化

- 在写代码时，Tensor 的维度形状变化是尤其需要关注的地方，理清楚逻辑可以减少遇到相关报错的可能
- 不同库和框架之间，相同 API 命名背后的实现可能完全不同  

## 问题思考

旧问题的解决往往伴随着新问题的诞生，让我们一起来思考一下：

- 我们在标记和收集数据的时候，特征的选取是否科学，数据集的规模应该要有多大？数据集的划分又应该选用什么样的比例？
- 这次见着了 `batch_size`, 它的设定是否有经验可循？接触到的超参数越来越多了，验证集和开发集是如何帮助我们对合适的超参数进行选取的？
- 我们对 NumPy 支持的 ndarray 的数据如何导入 MegEngine 已经有了经验，更复杂的数据集（图片、视频、音频）要如何处理呢？
- 噪声数据会对参数的学习进行干扰，`Dataloader` 中是否有对应的预处理方法来做一些统计意义上的处理？
- 我们知道了向量化可以加速运算，在硬件底层是如何实现“加速”这一效果的？

### 关于数学表示的习惯

由于数据矩阵 $X$ 的形状排布为 $(n, d)$, 因此 MegEngine 中实现的矩阵形式的线性回归为 $\hat {Y} = f(X) =  XW+B$ （这里将 $B$ 视为 $b$ 广播后的矩阵）

然而在线性代数中为了方便运算表示，向量通常默认为列向量 $\mathbf {x} = (x_1; x_2; \ldots x_n)$, 单样本线性回归为 $f(\mathbf {x};\mathbf {w},b) = \mathbf {w}^T\mathbf {x} + b$，

对应有形状为 $(d, n)$ 的数据矩阵：

<svg xmlns="http://www.w3.org/2000/svg" width="50.506ex" height="13.261ex" role="img" focusable="false" viewBox="0 -3180.8 22323.7 5861.5" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" style="vertical-align: -6.065ex;"><defs><path id="MJX-69-TEX-I-1D44B" d="M42 0H40Q26 0 26 11Q26 15 29 27Q33 41 36 43T55 46Q141 49 190 98Q200 108 306 224T411 342Q302 620 297 625Q288 636 234 637H206Q200 643 200 645T202 664Q206 677 212 683H226Q260 681 347 681Q380 681 408 681T453 682T473 682Q490 682 490 671Q490 670 488 658Q484 643 481 640T465 637Q434 634 411 620L488 426L541 485Q646 598 646 610Q646 628 622 635Q617 635 609 637Q594 637 594 648Q594 650 596 664Q600 677 606 683H618Q619 683 643 683T697 681T738 680Q828 680 837 683H845Q852 676 852 672Q850 647 840 637H824Q790 636 763 628T722 611T698 593L687 584Q687 585 592 480L505 384Q505 383 536 304T601 142T638 56Q648 47 699 46Q734 46 734 37Q734 35 732 23Q728 7 725 4T711 1Q708 1 678 1T589 2Q528 2 496 2T461 1Q444 1 444 10Q444 11 446 25Q448 35 450 39T455 44T464 46T480 47T506 54Q523 62 523 64Q522 64 476 181L429 299Q241 95 236 84Q232 76 232 72Q232 53 261 47Q262 47 267 47T273 46Q276 46 277 46T280 45T283 42T284 35Q284 26 282 19Q279 6 276 4T261 1Q258 1 243 1T201 2T142 2Q64 2 42 0Z"></path><path id="MJX-69-TEX-N-3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path><path id="MJX-69-TEX-S4-23A1" d="M319 -645V1154H666V1070H403V-645H319Z"></path><path id="MJX-69-TEX-S4-23A3" d="M319 -644V1155H403V-560H666V-644H319Z"></path><path id="MJX-69-TEX-S4-23A2" d="M319 0V602H403V0H319Z"></path><path id="MJX-69-TEX-N-7C" d="M139 -249H137Q125 -249 119 -235V251L120 737Q130 750 139 750Q152 750 159 735V-235Q151 -249 141 -249H139Z"></path><path id="MJX-69-TEX-B-1D431" d="M227 0Q212 3 121 3Q40 3 28 0H21V62H117L245 213L109 382H26V444H34Q49 441 143 441Q247 441 265 444H274V382H246L281 339Q315 297 316 297Q320 297 354 341L389 382H352V444H360Q375 441 466 441Q547 441 559 444H566V382H471L355 246L504 63L545 62H586V0H578Q563 3 469 3Q365 3 347 0H338V62H366Q366 63 326 112T285 163L198 63L217 62H235V0H227Z"></path><path id="MJX-69-TEX-N-31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path><path id="MJX-69-TEX-N-32" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z"></path><path id="MJX-69-TEX-N-22EF" d="M78 250Q78 274 95 292T138 310Q162 310 180 294T199 251Q199 226 182 208T139 190T96 207T78 250ZM525 250Q525 274 542 292T585 310Q609 310 627 294T646 251Q646 226 629 208T586 190T543 207T525 250ZM972 250Q972 274 989 292T1032 310Q1056 310 1074 294T1093 251Q1093 226 1076 208T1033 190T990 207T972 250Z"></path><path id="MJX-69-TEX-I-1D45B" d="M21 287Q22 293 24 303T36 341T56 388T89 425T135 442Q171 442 195 424T225 390T231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336T465 179T427 52Q427 26 444 26Q450 26 453 27Q482 32 505 65T540 145Q542 153 560 153Q580 153 580 145Q580 144 576 130Q568 101 554 73T508 17T439 -10Q392 -10 371 17T350 73Q350 92 386 193T423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 180T152 343Q153 348 153 366Q153 405 129 405Q91 405 66 305Q60 285 60 284Q58 278 41 278H27Q21 284 21 287Z"></path><path id="MJX-69-TEX-S4-23A4" d="M0 1070V1154H347V-645H263V1070H0Z"></path><path id="MJX-69-TEX-S4-23A6" d="M263 -560V1155H347V-644H0V-560H263Z"></path><path id="MJX-69-TEX-S4-23A5" d="M263 0V602H347V0H263Z"></path><path id="MJX-69-TEX-I-1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z"></path><path id="MJX-69-TEX-N-2C" d="M78 35T78 60T94 103T137 121Q165 121 187 96T210 8Q210 -27 201 -60T180 -117T154 -158T130 -185T117 -194Q113 -194 104 -185T95 -172Q95 -168 106 -156T131 -126T157 -76T173 -3V9L172 8Q170 7 167 6T161 3T152 1T140 0Q113 0 96 17Z"></path><path id="MJX-69-TEX-N-22EE" d="M78 30Q78 54 95 72T138 90Q162 90 180 74T199 31Q199 6 182 -12T139 -30T96 -13T78 30ZM78 440Q78 464 95 482T138 500Q162 500 180 484T199 441Q199 416 182 398T139 380T96 397T78 440ZM78 840Q78 864 95 882T138 900Q162 900 180 884T199 841Q199 816 182 798T139 780T96 797T78 840Z"></path><path id="MJX-69-TEX-I-1D451" d="M366 683Q367 683 438 688T511 694Q523 694 523 686Q523 679 450 384T375 83T374 68Q374 26 402 26Q411 27 422 35Q443 55 463 131Q469 151 473 152Q475 153 483 153H487H491Q506 153 506 145Q506 140 503 129Q490 79 473 48T445 8T417 -8Q409 -10 393 -10Q359 -10 336 5T306 36L300 51Q299 52 296 50Q294 48 292 46Q233 -10 172 -10Q117 -10 75 30T33 157Q33 205 53 255T101 341Q148 398 195 420T280 442Q336 442 364 400Q369 394 369 396Q370 400 396 505T424 616Q424 629 417 632T378 637H357Q351 643 351 645T353 664Q358 683 366 683ZM352 326Q329 405 277 405Q242 405 210 374T160 293Q131 214 119 129Q119 126 119 118T118 106Q118 61 136 44T179 26Q233 26 290 98L298 109L352 326Z"></path></defs><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><g data-mml-node="math"><g data-mml-node="mtable"><g data-mml-node="mtr"><g data-mml-node="mtd"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D44B"></use></g><g data-mml-node="mo" transform="translate(1129.8, 0)"><use xlink:href="#MJX-69-TEX-N-3D"></use></g><g data-mml-node="mrow" transform="translate(2185.6, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-S4-23A1" transform="translate(0, 996)"></use><use xlink:href="#MJX-69-TEX-S4-23A3" transform="translate(0, -1006)"></use><svg width="667" height="402" y="49" x="0" viewBox="0 100.5 667 402"><use xlink:href="#MJX-69-TEX-S4-23A2" transform="scale(1, 1.002)"></use></svg></g><g data-mml-node="mtable" transform="translate(667, 0)"><g data-mml-node="mtr" transform="translate(0, 1400)"><g data-mml-node="mtd" transform="translate(366.3, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-7C"></use></g></g></g><g data-mml-node="mtd" transform="translate(2376.8, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-7C"></use></g></g></g><g data-mml-node="mtd" transform="translate(4607.1, 0)"></g><g data-mml-node="mtd" transform="translate(6594.7, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-7C"></use></g></g></g></g><g data-mml-node="mtr"><g data-mml-node="mtd"><g data-mml-node="msub"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-B-1D431"></use></g></g><g data-mml-node="mn" transform="translate(607, -150) scale(0.707)"><use xlink:href="#MJX-69-TEX-N-31"></use></g></g></g><g data-mml-node="mtd" transform="translate(2010.6, 0)"><g data-mml-node="msub"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-B-1D431"></use></g></g><g data-mml-node="mn" transform="translate(607, -150) scale(0.707)"><use xlink:href="#MJX-69-TEX-N-32"></use></g></g></g><g data-mml-node="mtd" transform="translate(4021.1, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(6193.1, 0)"><g data-mml-node="msub"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-B-1D431"></use></g></g><g data-mml-node="mi" transform="translate(607, -150) scale(0.707)"><use xlink:href="#MJX-69-TEX-I-1D45B"></use></g></g></g></g><g data-mml-node="mtr" transform="translate(0, -1400)"><g data-mml-node="mtd" transform="translate(366.3, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-7C"></use></g></g></g><g data-mml-node="mtd" transform="translate(2376.8, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-7C"></use></g></g></g><g data-mml-node="mtd" transform="translate(4607.1, 0)"></g><g data-mml-node="mtd" transform="translate(6594.7, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-7C"></use></g></g></g></g></g><g data-mml-node="mo" transform="translate(7941.4, 0)"><use xlink:href="#MJX-69-TEX-S4-23A4" transform="translate(0, 996)"></use><use xlink:href="#MJX-69-TEX-S4-23A6" transform="translate(0, -1006)"></use><svg width="667" height="402" y="49" x="0" viewBox="0 100.5 667 402"><use xlink:href="#MJX-69-TEX-S4-23A5" transform="scale(1, 1.002)"></use></svg></g></g><g data-mml-node="mo" transform="translate(11071.7, 0)"><use xlink:href="#MJX-69-TEX-N-3D"></use></g><g data-mml-node="mrow" transform="translate(12127.5, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-S4-23A1" transform="translate(0, 2026.8)"></use><use xlink:href="#MJX-69-TEX-S4-23A3" transform="translate(0, -2036.8)"></use><svg width="667" height="2463.5" y="-981.8" x="0" viewBox="0 615.9 667 2463.5"><use xlink:href="#MJX-69-TEX-S4-23A2" transform="scale(1, 6.138)"></use></svg></g><g data-mml-node="mtable" transform="translate(667, 0)"><g data-mml-node="mtr" transform="translate(0, 2430.8)"><g data-mml-node="mtd" transform="translate(7.1, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-69-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-69-TEX-N-31"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(2546.9, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-69-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-69-TEX-N-32"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(5079.6, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(7258.7, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-69-TEX-N-31"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mi" transform="translate(778, 0)"><use xlink:href="#MJX-69-TEX-I-1D45B"></use></g></g></g></g></g><g data-mml-node="mtr" transform="translate(0, 993.6)"><g data-mml-node="mtd" transform="translate(7.1, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-69-TEX-N-32"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-69-TEX-N-31"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(2546.9, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-69-TEX-N-32"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(778, 0)"><use xlink:href="#MJX-69-TEX-N-32"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(5079.6, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(7258.7, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><use xlink:href="#MJX-69-TEX-N-32"></use></g><g data-mml-node="mo" transform="translate(500, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mi" transform="translate(778, 0)"><use xlink:href="#MJX-69-TEX-I-1D45B"></use></g></g></g></g></g><g data-mml-node="mtr" transform="translate(0, -993.6)"><g data-mml-node="mtd" transform="translate(630.9, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-22EE"></use></g></g></g><g data-mml-node="mtd" transform="translate(3309.7, 0)"></g><g data-mml-node="mtd" transform="translate(5665.6, 0)"></g><g data-mml-node="mtd" transform="translate(7917.9, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-22EE"></use></g></g></g></g><g data-mml-node="mtr" transform="translate(0, -2393.6)"><g data-mml-node="mtd"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D451"></use></g><g data-mml-node="mo" transform="translate(520, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(798, 0)"><use xlink:href="#MJX-69-TEX-N-31"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(2539.8, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D451"></use></g><g data-mml-node="mo" transform="translate(520, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mn" transform="translate(798, 0)"><use xlink:href="#MJX-69-TEX-N-32"></use></g></g></g></g><g data-mml-node="mtd" transform="translate(5079.6, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-69-TEX-N-22EF"></use></g></g><g data-mml-node="mtd" transform="translate(7251.6, 0)"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D465"></use></g><g data-mml-node="TeXAtom" transform="translate(572, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-69-TEX-I-1D451"></use></g><g data-mml-node="mo" transform="translate(520, 0)"><use xlink:href="#MJX-69-TEX-N-2C"></use></g><g data-mml-node="mi" transform="translate(798, 0)"><use xlink:href="#MJX-69-TEX-I-1D45B"></use></g></g></g></g></g></g><g data-mml-node="mo" transform="translate(9529.2, 0)"><use xlink:href="#MJX-69-TEX-S4-23A4" transform="translate(0, 2026.8)"></use><use xlink:href="#MJX-69-TEX-S4-23A6" transform="translate(0, -2036.8)"></use><svg width="667" height="2463.5" y="-981.8" x="0" viewBox="0 615.9 667 2463.5"><use xlink:href="#MJX-69-TEX-S4-23A5" transform="scale(1, 6.138)"></use></svg></g></g></g></g></g></g></g></svg>

因此可以得到 $\hat {Y} = f(X) =  W^{T}X+B$, 这反而是比较常见的形式（类似于 $y=Ax+b$）。

为什么我们在教程中要使用 $\hat {Y} = f(X) = XW+B$ 这种不常见的表述形式呢？数据布局不同会有什么影响？

深度学习，简单开发。我们鼓励你在实践中不断思考，并启发自己去探索直觉性或理论性的解释。

本文章原文链接：[https://megengine.org.cn/doc/stable/zh/getting-started/beginner/learning-from-linear-regression.html](https://megengine.org.cn/doc/stable/zh/getting-started/beginner/learning-from-linear-regression.html?id=NumpyLinear_page2_docstart_01)
