# %%
import numpy as np
import matplotlib.pyplot as plt

input_text = """epoch: 0 loss: 2520.820051074028
Train accuracy of the network on the 100000 diagrams: 21.22900 %
Validation accuracy of the network on the 100000 diagrams: 21.29200 %
epoch: 1 loss: 2498.606311917305
epoch: 2 loss: 1710.5863739848137
epoch: 3 loss: 575.6613089591265
epoch: 4 loss: 523.9547325223684
epoch: 5 loss: 492.58750370144844
epoch: 6 loss: 474.0783798992634
epoch: 7 loss: 465.9730302467942
epoch: 8 loss: 469.73896907269955
epoch: 9 loss: 453.7575627043843
epoch: 10 loss: 443.5889809951186
Train accuracy of the network on the 100000 diagrams: 89.31800 %
Validation accuracy of the network on the 100000 diagrams: 89.34700 %
epoch: 11 loss: 444.285699672997
epoch: 12 loss: 434.0318601205945
epoch: 13 loss: 450.06854002922773
epoch: 14 loss: 425.7892456948757
epoch: 15 loss: 422.65797543525696
epoch: 16 loss: 424.3853740617633
epoch: 17 loss: 420.2684548199177
epoch: 18 loss: 417.4620707407594
epoch: 19 loss: 412.72124322503805
epoch: 20 loss: 414.18170569837093
Train accuracy of the network on the 100000 diagrams: 90.28500 %
Validation accuracy of the network on the 100000 diagrams: 90.34600 %
epoch: 21 loss: 412.37111124768853
epoch: 22 loss: 411.56928527355194
epoch: 23 loss: 413.43362349271774
epoch: 24 loss: 405.22066543996334
epoch: 25 loss: 404.64106998592615
epoch: 26 loss: 402.7278224825859
epoch: 27 loss: 408.1373560875654
epoch: 28 loss: 399.35307324677706
epoch: 29 loss: 399.912629686296
epoch: 30 loss: 403.7230020239949
Train accuracy of the network on the 100000 diagrams: 90.11000 %
Validation accuracy of the network on the 100000 diagrams: 90.07200 %
epoch: 31 loss: 410.8136212155223
epoch: 32 loss: 399.9528777003288
epoch: 33 loss: 397.7707670480013
epoch: 34 loss: 396.45055598020554
epoch: 35 loss: 397.6692738905549
epoch: 36 loss: 394.5345527231693
epoch: 37 loss: 390.968311175704
epoch: 38 loss: 396.9733263924718
epoch: 39 loss: 392.4250451475382
epoch: 40 loss: 393.8597699776292
Train accuracy of the network on the 100000 diagrams: 89.87300 %
Validation accuracy of the network on the 100000 diagrams: 89.81900 %
epoch: 41 loss: 390.336730889976
epoch: 42 loss: 390.0413496643305
epoch: 43 loss: 389.47009091824293
epoch: 44 loss: 387.7974504083395
epoch: 45 loss: 403.0253709591925
epoch: 46 loss: 382.25044221431017
epoch: 47 loss: 388.8362224176526
epoch: 48 loss: 385.44676945358515
epoch: 49 loss: 387.3399634733796
epoch: 50 loss: 384.2032771706581
Train accuracy of the network on the 100000 diagrams: 90.41300 %
Validation accuracy of the network on the 100000 diagrams: 90.55200 %
epoch: 51 loss: 387.69889922067523
epoch: 52 loss: 382.91975420713425
epoch: 53 loss: 382.9716539606452
epoch: 54 loss: 382.0116897597909
epoch: 55 loss: 380.75741128623486
epoch: 56 loss: 387.18996649235487
epoch: 57 loss: 378.149312607944
epoch: 58 loss: 385.6769947335124
epoch: 59 loss: 391.55822851508856
epoch: 60 loss: 381.0091728940606
Train accuracy of the network on the 100000 diagrams: 90.22900 %
Validation accuracy of the network on the 100000 diagrams: 90.32300 %
epoch: 61 loss: 378.6386247500777
epoch: 62 loss: 380.56330543011427
epoch: 63 loss: 377.63786079734564
epoch: 64 loss: 376.15439853817225
epoch: 65 loss: 380.63283386081457
epoch: 66 loss: 381.81957622617483
epoch: 67 loss: 377.41289434581995
epoch: 68 loss: 377.3433849848807
epoch: 69 loss: 379.8768713772297
epoch: 70 loss: 379.52656201273203
Train accuracy of the network on the 100000 diagrams: 90.65900 %
Validation accuracy of the network on the 100000 diagrams: 90.63000 %
epoch: 71 loss: 375.2173762768507
epoch: 72 loss: 382.9036213606596
epoch: 73 loss: 371.49840262532234
epoch: 74 loss: 373.71984430775046
epoch: 75 loss: 379.2100046463311
epoch: 76 loss: 373.5215090960264
epoch: 77 loss: 376.7132279165089
epoch: 78 loss: 373.4540106728673
epoch: 79 loss: 374.60747496783733
epoch: 80 loss: 370.6838428005576
Train accuracy of the network on the 100000 diagrams: 90.63300 %
Validation accuracy of the network on the 100000 diagrams: 90.62200 %
epoch: 81 loss: 371.4450107216835
epoch: 82 loss: 373.6522675231099
epoch: 83 loss: 369.1787295117974
epoch: 84 loss: 372.4226860180497
epoch: 85 loss: 380.81964315474033
epoch: 86 loss: 368.9289445243776
epoch: 87 loss: 372.1175679266453
epoch: 88 loss: 369.28662572056055
epoch: 89 loss: 368.8562491238117
epoch: 90 loss: 369.74156649783254
Train accuracy of the network on the 100000 diagrams: 90.69300 %
Validation accuracy of the network on the 100000 diagrams: 90.66100 %
epoch: 91 loss: 370.633344065398
epoch: 92 loss: 368.21595134958625
epoch: 93 loss: 368.4260578751564
epoch: 94 loss: 370.1547223776579
epoch: 95 loss: 366.8629308268428
epoch: 96 loss: 371.1013488397002
epoch: 97 loss: 366.8103491291404
epoch: 98 loss: 368.9669823423028
epoch: 99 loss: 366.6638095378876
epoch: 100 loss: 367.2313104271889
Train accuracy of the network on the 100000 diagrams: 91.07700 %
Validation accuracy of the network on the 100000 diagrams: 91.05100 %
epoch: 101 loss: 366.81751162931323
epoch: 102 loss: 380.2187173888087
epoch: 103 loss: 364.5425571091473
epoch: 104 loss: 367.22126864269376
epoch: 105 loss: 375.4177265241742
epoch: 106 loss: 367.24864596128464
epoch: 107 loss: 364.6665424630046
epoch: 108 loss: 365.39274222403765
epoch: 109 loss: 371.6963750272989
epoch: 110 loss: 364.4734148234129
Train accuracy of the network on the 100000 diagrams: 91.04200 %
Validation accuracy of the network on the 100000 diagrams: 90.94800 %
epoch: 111 loss: 365.76808566227555
epoch: 112 loss: 370.0328317582607
epoch: 113 loss: 365.0561785623431
epoch: 114 loss: 367.8530347943306
epoch: 115 loss: 370.32047229260206
epoch: 116 loss: 362.42435777559876
epoch: 117 loss: 363.8203718550503
epoch: 118 loss: 365.0577178224921
epoch: 119 loss: 364.75742146372795
epoch: 120 loss: 365.16357785463333
Train accuracy of the network on the 100000 diagrams: 90.68800 %
Validation accuracy of the network on the 100000 diagrams: 90.58300 %
epoch: 121 loss: 363.1830997541547
epoch: 122 loss: 363.8051905333996
epoch: 123 loss: 376.34099108725786
epoch: 124 loss: 361.03714713826776
epoch: 125 loss: 360.28185215592384
epoch: 126 loss: 360.84850024804473
epoch: 127 loss: 360.5275559425354
epoch: 128 loss: 360.94546864554286
epoch: 129 loss: 365.43102114647627
epoch: 130 loss: 360.07733937352896
Train accuracy of the network on the 100000 diagrams: 90.76400 %
Validation accuracy of the network on the 100000 diagrams: 90.86900 %
epoch: 131 loss: 361.56301010772586
epoch: 132 loss: 369.6944389604032
epoch: 133 loss: 359.0336563587189
epoch: 134 loss: 366.051553376019
epoch: 135 loss: 358.42294012755156
epoch: 136 loss: 365.5910128057003
epoch: 137 loss: 358.9352806136012
epoch: 138 loss: 358.7532711699605
epoch: 139 loss: 361.0897631496191
epoch: 140 loss: 364.0882465913892
Train accuracy of the network on the 100000 diagrams: 91.09000 %
Validation accuracy of the network on the 100000 diagrams: 91.04100 %
epoch: 141 loss: 358.57168141007423
epoch: 142 loss: 360.2410276532173
epoch: 143 loss: 358.06132209673524
epoch: 144 loss: 363.4711475595832
epoch: 145 loss: 355.0508386082947
epoch: 146 loss: 357.44658094644547
epoch: 147 loss: 360.48383086919785
epoch: 148 loss: 361.0472703203559
epoch: 149 loss: 359.0473988354206
epoch: 150 loss: 358.0700492449105
Train accuracy of the network on the 100000 diagrams: 91.14500 %
Validation accuracy of the network on the 100000 diagrams: 91.16700 %
epoch: 151 loss: 359.49258260428905
epoch: 152 loss: 359.01991933584213
epoch: 153 loss: 356.0498683862388
epoch: 154 loss: 357.46695432439446
epoch: 155 loss: 356.7863223068416
epoch: 156 loss: 359.86585734784603
epoch: 157 loss: 356.7293435111642
epoch: 158 loss: 357.3001644909382
epoch: 159 loss: 358.0596433132887
epoch: 160 loss: 369.8577077537775
Train accuracy of the network on the 100000 diagrams: 91.21300 %
Validation accuracy of the network on the 100000 diagrams: 91.22000 %
epoch: 161 loss: 377.0622475259006
epoch: 162 loss: 355.5548691675067
epoch: 163 loss: 368.2440253607929
epoch: 164 loss: 352.8382074609399
epoch: 165 loss: 356.3379524573684
epoch: 166 loss: 362.8373056128621
epoch: 167 loss: 356.58824095875025
epoch: 168 loss: 377.4560771211982
epoch: 169 loss: 377.93577632680535
epoch: 170 loss: 353.4881820715964
Train accuracy of the network on the 100000 diagrams: 90.85100 %
Validation accuracy of the network on the 100000 diagrams: 90.86400 %
epoch: 171 loss: 355.7854686677456
epoch: 172 loss: 358.90881738066673
epoch: 173 loss: 358.7313284948468
epoch: 174 loss: 356.0960435792804
epoch: 175 loss: 356.9282948896289
epoch: 176 loss: 354.71638052910566
epoch: 177 loss: 362.9247125387192
epoch: 178 loss: 367.3456748351455
epoch: 179 loss: 354.8914794996381
epoch: 180 loss: 352.6900393515825
Train accuracy of the network on the 100000 diagrams: 91.26200 %
Validation accuracy of the network on the 100000 diagrams: 91.32100 %
epoch: 181 loss: 363.5187188051641
epoch: 182 loss: 353.9486335553229
epoch: 183 loss: 355.20345137268305
epoch: 184 loss: 356.0933549180627
epoch: 185 loss: 356.2186306193471
epoch: 186 loss: 360.58173445239663
epoch: 187 loss: 354.1022971943021
epoch: 188 loss: 355.1100058853626
epoch: 189 loss: 355.54558453708887
epoch: 190 loss: 353.63587963953614
Train accuracy of the network on the 100000 diagrams: 91.07200 %
Validation accuracy of the network on the 100000 diagrams: 91.09800 %
epoch: 191 loss: 357.4596046805382
epoch: 192 loss: 356.52036997675896
epoch: 193 loss: 354.4006474688649
epoch: 194 loss: 357.16232542693615
epoch: 195 loss: 353.3053761534393
epoch: 196 loss: 354.23758083954453
epoch: 197 loss: 380.6598278656602
epoch: 198 loss: 350.77764519304037
epoch: 199 loss: 360.00158336386085
epoch: 200 loss: 357.65967078506947
Train accuracy of the network on the 100000 diagrams: 91.36900 %
Validation accuracy of the network on the 100000 diagrams: 91.43600 %
epoch: 201 loss: 352.6692701205611
epoch: 202 loss: 355.2230293825269
epoch: 203 loss: 350.8765818551183
epoch: 204 loss: 362.5204011425376
epoch: 205 loss: 358.4281073920429
epoch: 206 loss: 374.4982080757618
epoch: 207 loss: 350.42483992129564
epoch: 208 loss: 352.20741795375943
epoch: 209 loss: 360.7556967139244
epoch: 210 loss: 352.6603974029422
Train accuracy of the network on the 100000 diagrams: 90.93500 %
Validation accuracy of the network on the 100000 diagrams: 90.83900 %
epoch: 211 loss: 354.4403009302914
epoch: 212 loss: 354.052681311965
epoch: 214 loss: 373.72952683642507
epoch: 215 loss: 351.89909121766686
epoch: 216 loss: 351.33947080373764
epoch: 217 loss: 354.4611507989466
epoch: 218 loss: 354.4644593074918
epoch: 219 loss: 352.08886467665434
epoch: 220 loss: 354.14789049327374
Train accuracy of the network on the 100000 diagrams: 91.27200 %
Validation accuracy of the network on the 100000 diagrams: 91.31200 %
epoch: 221 loss: 352.8010999709368
epoch: 222 loss: 354.33351930975914
epoch: 223 loss: 364.3584346398711
epoch: 224 loss: 350.7800609767437
epoch: 225 loss: 351.9498697705567
epoch: 226 loss: 355.0358531624079
epoch: 227 loss: 353.6827112175524
epoch: 228 loss: 355.39254804700613
epoch: 229 loss: 357.06499273329973
epoch: 230 loss: 353.3648697659373
Train accuracy of the network on the 100000 diagrams: 90.99600 %
Validation accuracy of the network on the 100000 diagrams: 91.00000 %
epoch: 231 loss: 358.8808047026396
epoch: 232 loss: 348.8550102189183
epoch: 233 loss: 359.679535318166
epoch: 234 loss: 359.6466013751924
epoch: 235 loss: 349.88908487930894
epoch: 236 loss: 352.1950700394809
epoch: 237 loss: 353.22461408004165
epoch: 238 loss: 359.0547160357237
epoch: 239 loss: 349.9700390249491
epoch: 240 loss: 353.3493359014392
Train accuracy of the network on the 100000 diagrams: 91.11400 %
Validation accuracy of the network on the 100000 diagrams: 91.18000 %
epoch: 241 loss: 355.6682714447379
epoch: 242 loss: 349.578991137445
epoch: 243 loss: 351.2405927889049
epoch: 244 loss: 354.1534376591444
epoch: 245 loss: 350.1291777230799
epoch: 246 loss: 350.58635625988245
epoch: 247 loss: 357.17476765438914
epoch: 248 loss: 351.6635914221406
epoch: 249 loss: 350.5193904340267
epoch: 250 loss: 372.7271465025842
Train accuracy of the network on the 100000 diagrams: 90.07200 %
Validation accuracy of the network on the 100000 diagrams: 90.08000 %
epoch: 251 loss: 352.016211040318
epoch: 252 loss: 349.25499991327524
epoch: 253 loss: 351.7851623147726
epoch: 254 loss: 349.2457894310355
epoch: 255 loss: 352.53715182095766
epoch: 256 loss: 355.1820090226829
epoch: 257 loss: 349.95402456074953
epoch: 258 loss: 352.7550215870142
epoch: 259 loss: 351.64220029860735
epoch: 260 loss: 355.0882723554969
Train accuracy of the network on the 100000 diagrams: 90.95400 %
Validation accuracy of the network on the 100000 diagrams: 90.83300 %
epoch: 261 loss: 348.4701778329909
epoch: 262 loss: 349.8379774540663
epoch: 263 loss: 351.8201530352235
epoch: 264 loss: 349.2593453153968
epoch: 265 loss: 359.6313615143299
epoch: 266 loss: 353.0400133654475
epoch: 267 loss: 347.8667254820466
epoch: 268 loss: 356.0353492349386
epoch: 269 loss: 347.9080585464835
epoch: 270 loss: 356.3651595711708
Train accuracy of the network on the 100000 diagrams: 91.08900 %
Validation accuracy of the network on the 100000 diagrams: 91.05300 %
epoch: 271 loss: 348.7586658895016
epoch: 272 loss: 351.0179184488952
epoch: 273 loss: 350.41037530079484
epoch: 274 loss: 351.90914696455
epoch: 275 loss: 350.3431451097131
epoch: 276 loss: 359.89852990210056
epoch: 277 loss: 350.51555347442627
epoch: 278 loss: 352.02725975960493
epoch: 279 loss: 355.07961194217205
epoch: 280 loss: 351.91518807038665
Train accuracy of the network on the 100000 diagrams: 91.09500 %
Validation accuracy of the network on the 100000 diagrams: 90.99800 %
epoch: 281 loss: 348.03282106667757
epoch: 282 loss: 349.5542921125889
epoch: 283 loss: 354.277249597013
epoch: 284 loss: 348.92432487010956
epoch: 285 loss: 349.6327446885407
epoch: 286 loss: 350.1651931926608
epoch: 287 loss: 347.5627936422825
epoch: 288 loss: 348.0586530342698
epoch: 289 loss: 349.40155639126897
epoch: 290 loss: 348.75044640898705
Train accuracy of the network on the 100000 diagrams: 90.77900 %
Validation accuracy of the network on the 100000 diagrams: 90.71800 %
epoch: 291 loss: 349.04794735088944
epoch: 292 loss: 352.2118833102286
epoch: 293 loss: 352.66154680401087
epoch: 294 loss: 345.9856736548245
epoch: 295 loss: 354.1738107651472
epoch: 296 loss: 348.79040468111634
epoch: 297 loss: 348.0264646150172
epoch: 298 loss: 351.82927372306585
epoch: 299 loss: 348.0440175011754
epoch: 300 loss: 345.12434523925185
Train accuracy of the network on the 100000 diagrams: 91.10500 %
Validation accuracy of the network on the 100000 diagrams: 91.03600 %
epoch: 301 loss: 365.5387507379055
epoch: 302 loss: 346.5856770500541
epoch: 303 loss: 347.36253217607737
epoch: 304 loss: 348.7018464244902
epoch: 305 loss: 358.6105421036482
epoch: 306 loss: 352.47339986264706
epoch: 307 loss: 344.7714690156281
epoch: 308 loss: 352.46287635713816
epoch: 309 loss: 346.40865985676646
epoch: 310 loss: 349.154071316123
Train accuracy of the network on the 100000 diagrams: 90.55600 %
Validation accuracy of the network on the 100000 diagrams: 90.56400 %
epoch: 311 loss: 351.0070597752929
epoch: 312 loss: 347.30696660652757
epoch: 313 loss: 362.33598978072405
epoch: 314 loss: 344.6265297345817
epoch: 315 loss: 347.0345590226352
epoch: 316 loss: 350.97363363578916
epoch: 317 loss: 346.08209438622
epoch: 319 loss: 346.6817026697099
epoch: 320 loss: 348.3943953998387
Train accuracy of the network on the 100000 diagrams: 91.03800 %
Validation accuracy of the network on the 100000 diagrams: 91.00300 %
epoch: 321 loss: 541.24978543818
epoch: 322 loss: 359.33498767763376
epoch: 323 loss: 369.8245045952499
epoch: 324 loss: 356.727444447577
epoch: 325 loss: 356.23964313045144
epoch: 326 loss: 350.1412109658122
epoch: 327 loss: 350.2942601889372
epoch: 328 loss: 357.1649890616536
epoch: 329 loss: 345.9860022328794
epoch: 330 loss: 352.4255278073251
Train accuracy of the network on the 100000 diagrams: 90.91100 %
Validation accuracy of the network on the 100000 diagrams: 90.90100 %
epoch: 331 loss: 345.65356067195535
epoch: 332 loss: 371.81179213523865
epoch: 333 loss: 358.45144329965115
epoch: 334 loss: 380.6362984329462
epoch: 335 loss: 346.12419207394123
epoch: 336 loss: 378.1046939045191
epoch: 337 loss: 348.8679424673319
epoch: 338 loss: 377.39700895547867
epoch: 339 loss: 348.25134006887674
epoch: 340 loss: 349.35847505554557
Train accuracy of the network on the 100000 diagrams: 90.98000 %
Validation accuracy of the network on the 100000 diagrams: 90.93100 %
epoch: 341 loss: 347.1725808829069
epoch: 342 loss: 349.0651606991887
epoch: 343 loss: 348.9454402178526
epoch: 344 loss: 345.0475938990712
epoch: 345 loss: 350.1894930265844
epoch: 346 loss: 346.2369047142565
epoch: 347 loss: 346.7548157274723
epoch: 348 loss: 348.40213872492313
epoch: 349 loss: 348.503661878407
epoch: 350 loss: 344.6805932931602
Train accuracy of the network on the 100000 diagrams: 91.40900 %
Validation accuracy of the network on the 100000 diagrams: 91.30800 %
epoch: 351 loss: 346.54635805264115
epoch: 352 loss: 346.0132972970605
epoch: 353 loss: 366.8997928202152
epoch: 354 loss: 344.8574969172478
epoch: 355 loss: 349.83591113239527
epoch: 356 loss: 345.2184951454401
epoch: 357 loss: 347.26660515740514
epoch: 358 loss: 345.47875664010644
epoch: 359 loss: 344.711538400501
epoch: 360 loss: 368.4190600588918
Train accuracy of the network on the 100000 diagrams: 91.34200 %
Validation accuracy of the network on the 100000 diagrams: 91.41100 %
epoch: 361 loss: 355.08649883046746
epoch: 362 loss: 427.1987060941756
epoch: 363 loss: 376.96937388181686
epoch: 364 loss: 346.1444612890482
epoch: 365 loss: 349.94487972930074
epoch: 366 loss: 396.74463368207216
epoch: 367 loss: 363.79300163686275
epoch: 368 loss: 351.0283377021551
epoch: 369 loss: 349.7990650460124
epoch: 370 loss: 348.1823721937835
Train accuracy of the network on the 100000 diagrams: 91.11200 %
Validation accuracy of the network on the 100000 diagrams: 91.03300 %
epoch: 371 loss: 349.47473138943315
epoch: 372 loss: 357.2996528968215
epoch: 373 loss: 345.6799061521888
epoch: 374 loss: 344.7872791737318
epoch: 375 loss: 345.9644960090518
epoch: 376 loss: 347.18947964161634
epoch: 377 loss: 349.9097246564925
epoch: 378 loss: 344.2781235370785
epoch: 379 loss: 359.80784720927477
epoch: 380 loss: 347.23047870770097
Train accuracy of the network on the 100000 diagrams: 91.43800 %
Validation accuracy of the network on the 100000 diagrams: 91.36400 %
epoch: 381 loss: 345.66287426277995
epoch: 382 loss: 345.41036761552095
epoch: 383 loss: 346.3285271897912
epoch: 384 loss: 346.7184575498104
epoch: 385 loss: 349.07938111945987
epoch: 386 loss: 351.6151118353009
epoch: 387 loss: 345.59356340393424
epoch: 388 loss: 344.7559348605573
epoch: 389 loss: 345.1625173166394
epoch: 390 loss: 346.64329497516155
Train accuracy of the network on the 100000 diagrams: 91.13800 %
Validation accuracy of the network on the 100000 diagrams: 90.99700 %
epoch: 391 loss: 346.07151063904166
epoch: 392 loss: 344.482841335237
epoch: 393 loss: 348.2300538048148
epoch: 394 loss: 343.36096158623695
epoch: 395 loss: 345.512194275856
epoch: 396 loss: 346.16459072753787
epoch: 397 loss: 342.44781682640314
epoch: 398 loss: 419.25767830759287
epoch: 399 loss: 362.35220455005765
epoch: 400 loss: 343.7239497676492
Train accuracy of the network on the 100000 diagrams: 91.17300 %
Validation accuracy of the network on the 100000 diagrams: 91.05000 %
epoch: 401 loss: 340.8593975827098
epoch: 402 loss: 345.24959636107087
epoch: 403 loss: 349.15889940038323
epoch: 404 loss: 344.4874094873667
epoch: 405 loss: 350.1324869841337
epoch: 406 loss: 344.4923827573657
epoch: 407 loss: 346.8934748917818
epoch: 408 loss: 361.9120718576014
epoch: 409 loss: 346.01315900683403
epoch: 410 loss: 342.9470938667655
Train accuracy of the network on the 100000 diagrams: 91.32000 %
Validation accuracy of the network on the 100000 diagrams: 91.15300 %
epoch: 411 loss: 343.31030575186014
epoch: 412 loss: 343.15904869511724
epoch: 413 loss: 365.99770148843527
epoch: 414 loss: 349.48245905712247
epoch: 415 loss: 343.21608236804605
epoch: 416 loss: 343.90176586434245
epoch: 417 loss: 346.1415617763996
epoch: 418 loss: 341.37928557395935
epoch: 419 loss: 346.63439305126667
epoch: 420 loss: 372.25797797739506
Train accuracy of the network on the 100000 diagrams: 91.54900 %
Validation accuracy of the network on the 100000 diagrams: 91.50300 %
epoch: 421 loss: 342.8675142079592
epoch: 422 loss: 357.8841999396682
epoch: 423 loss: 340.5195386633277
epoch: 424 loss: 342.6065193116665
epoch: 425 loss: 340.63080894201994
epoch: 426 loss: 344.7625529691577
epoch: 427 loss: 343.36287290230393
epoch: 428 loss: 475.6995187290013
epoch: 429 loss: 353.653963804245
epoch: 430 loss: 343.4848239980638
Train accuracy of the network on the 100000 diagrams: 91.15700 %
Validation accuracy of the network on the 100000 diagrams: 91.03300 %
epoch: 431 loss: 345.50811598822474
epoch: 432 loss: 344.6840584203601
epoch: 433 loss: 356.57720505818725
epoch: 434 loss: 344.2498341538012
epoch: 435 loss: 345.28001790866256
epoch: 436 loss: 359.1617344543338
epoch: 437 loss: 344.9176753498614
epoch: 438 loss: 341.27436262369156
epoch: 439 loss: 351.1921594366431
epoch: 440 loss: 341.65954134613276
Train accuracy of the network on the 100000 diagrams: 91.32200 %
Validation accuracy of the network on the 100000 diagrams: 91.23400 %
epoch: 441 loss: 344.9868494682014
epoch: 442 loss: 349.04664669185877
epoch: 443 loss: 343.6981216855347
epoch: 444 loss: 377.76795590668917
epoch: 445 loss: 340.8338341191411
epoch: 446 loss: 339.8026285097003
epoch: 447 loss: 359.8381000906229
epoch: 448 loss: 341.57917999848723
epoch: 449 loss: 343.93331887573004
epoch: 450 loss: 346.621815122664
Train accuracy of the network on the 100000 diagrams: 91.45000 %
Validation accuracy of the network on the 100000 diagrams: 91.37600 %
epoch: 451 loss: 341.8836631812155
epoch: 452 loss: 342.4363882429898
epoch: 453 loss: 341.6284932382405
epoch: 454 loss: 341.19233694300056
epoch: 455 loss: 342.95097110420465
epoch: 456 loss: 342.02702359855175
epoch: 457 loss: 352.8224429488182
epoch: 458 loss: 339.73615615069866
epoch: 459 loss: 372.07209853455424
epoch: 460 loss: 344.282912991941
Train accuracy of the network on the 100000 diagrams: 91.03300 %
Validation accuracy of the network on the 100000 diagrams: 90.98800 %
epoch: 461 loss: 339.2454273849726
epoch: 462 loss: 344.1978458017111
epoch: 463 loss: 342.2390930019319
epoch: 464 loss: 342.56815111264586
epoch: 465 loss: 342.64363545924425
epoch: 466 loss: 344.8691651523113
epoch: 467 loss: 340.9873791113496
epoch: 468 loss: 361.87845865637064
epoch: 469 loss: 339.73324348032475
epoch: 470 loss: 348.4098297730088
Train accuracy of the network on the 100000 diagrams: 90.86400 %
Validation accuracy of the network on the 100000 diagrams: 90.77600 %
epoch: 471 loss: 340.9811499901116
epoch: 472 loss: 338.0272388420999
epoch: 473 loss: 341.6288264282048
epoch: 474 loss: 356.42696437984705
epoch: 475 loss: 341.2825824059546
epoch: 476 loss: 341.46789744868875
epoch: 477 loss: 340.06009642779827
epoch: 478 loss: 339.94881051406264
epoch: 479 loss: 344.21732488274574
epoch: 480 loss: 341.73970130085945
Train accuracy of the network on the 100000 diagrams: 90.97900 %
Validation accuracy of the network on the 100000 diagrams: 90.96400 %
epoch: 481 loss: 338.7659278437495
epoch: 482 loss: 697.9374585002661
epoch: 483 loss: 401.5566865801811
epoch: 484 loss: 376.8818531483412
epoch: 485 loss: 366.48673225939274
epoch: 486 loss: 359.74776094406843
epoch: 487 loss: 354.83951779827476
epoch: 488 loss: 348.38298319652677
epoch: 489 loss: 389.75530568510294
epoch: 490 loss: 352.6780810058117
Train accuracy of the network on the 100000 diagrams: 91.18500 %
Validation accuracy of the network on the 100000 diagrams: 91.04300 %
epoch: 491 loss: 346.0948120728135
epoch: 492 loss: 349.62392046675086
epoch: 493 loss: 342.07711908966303
epoch: 494 loss: 344.31156615912914
epoch: 495 loss: 344.7346457503736
epoch: 496 loss: 342.9844827950001
epoch: 497 loss: 368.01302406936884
epoch: 498 loss: 340.91604306548834
epoch: 499 loss: 366.1760504245758
epoch: 500 loss: 343.35299644619226
Train accuracy of the network on the 100000 diagrams: 91.48800 %
Validation accuracy of the network on the 100000 diagrams: 91.40400 %
epoch: 501 loss: 340.2363770529628
epoch: 502 loss: 341.0980912670493
epoch: 503 loss: 345.0472643300891
epoch: 504 loss: 343.40744687989354
epoch: 505 loss: 340.77923441678286
epoch: 506 loss: 341.14084135368466
epoch: 507 loss: 341.1761060729623
epoch: 508 loss: 344.2413844540715
epoch: 509 loss: 339.4243463985622
epoch: 510 loss: 343.37612284719944
Train accuracy of the network on the 100000 diagrams: 91.47800 %
Validation accuracy of the network on the 100000 diagrams: 91.36300 %
epoch: 511 loss: 337.7356774173677
epoch: 512 loss: 342.2636595591903
epoch: 513 loss: 340.69448049366474
epoch: 514 loss: 338.8523127287626
epoch: 515 loss: 344.5329944342375
epoch: 516 loss: 337.5983796939254
epoch: 517 loss: 342.02130813896656
epoch: 518 loss: 345.91232201457024
epoch: 519 loss: 337.1830612272024
epoch: 520 loss: 433.8669550418854
Train accuracy of the network on the 100000 diagrams: 90.84800 %
Validation accuracy of the network on the 100000 diagrams: 90.75500 %
epoch: 521 loss: 340.58553037047386
epoch: 522 loss: 338.61554684117436
epoch: 523 loss: 340.42764714732766
epoch: 524 loss: 341.7246134504676
epoch: 525 loss: 340.4970239326358
epoch: 526 loss: 341.07341349869967
epoch: 527 loss: 340.25304935500026
epoch: 528 loss: 346.9615449607372
epoch: 529 loss: 339.1630767583847
epoch: 530 loss: 351.7150038331747
Train accuracy of the network on the 100000 diagrams: 91.55500 %
Validation accuracy of the network on the 100000 diagrams: 91.48600 %
epoch: 531 loss: 335.97966965660453
epoch: 532 loss: 338.43659200146794
epoch: 533 loss: 336.9528104402125
epoch: 534 loss: 338.8880504965782
epoch: 535 loss: 342.4601961709559
epoch: 536 loss: 338.9654743857682
epoch: 537 loss: 338.7070967592299
epoch: 538 loss: 343.12883538752794
epoch: 539 loss: 335.60207087546587
epoch: 540 loss: 338.2098067253828
Train accuracy of the network on the 100000 diagrams: 91.48600 %
Validation accuracy of the network on the 100000 diagrams: 91.37200 %
epoch: 541 loss: 340.82898669689894
epoch: 542 loss: 345.02185612916946
epoch: 543 loss: 343.202190708369
epoch: 544 loss: 338.7673945501447
epoch: 545 loss: 338.9015275016427
epoch: 546 loss: 350.73537093400955
epoch: 547 loss: 335.1788357309997
epoch: 548 loss: 337.3973470404744
epoch: 549 loss: 342.032975487411
epoch: 550 loss: 338.0876885764301
Train accuracy of the network on the 100000 diagrams: 91.50300 %
Validation accuracy of the network on the 100000 diagrams: 91.46700 %
epoch: 551 loss: 341.6842680349946
epoch: 552 loss: 339.17384992167354
epoch: 553 loss: 337.114843275398
epoch: 554 loss: 337.0345424488187
epoch: 555 loss: 338.54223757982254
epoch: 556 loss: 347.06723277270794
epoch: 557 loss: 337.7033515125513
epoch: 558 loss: 336.6416989713907
epoch: 559 loss: 343.2356937676668
epoch: 560 loss: 335.9632193259895
Train accuracy of the network on the 100000 diagrams: 91.57300 %
Validation accuracy of the network on the 100000 diagrams: 91.49600 %
epoch: 561 loss: 340.9282951429486
epoch: 562 loss: 343.62801998108625
epoch: 563 loss: 338.044460080564
epoch: 564 loss: 789.5605608448386
epoch: 565 loss: 464.33355379104614
epoch: 566 loss: 483.2499072700739
epoch: 567 loss: 451.22163734585047
epoch: 568 loss: 406.79957726597786
epoch: 569 loss: 374.33613800257444
epoch: 570 loss: 354.77164490520954
Train accuracy of the network on the 100000 diagrams: 91.24900 %
Validation accuracy of the network on the 100000 diagrams: 91.24700 %
epoch: 571 loss: 351.8483739234507
epoch: 572 loss: 398.20857349038124
epoch: 573 loss: 351.8715298175812
epoch: 574 loss: 357.1116053760052
epoch: 575 loss: 373.9458822235465
epoch: 576 loss: 406.9592084363103
epoch: 577 loss: 356.05136701092124
epoch: 578 loss: 349.4242833070457
epoch: 579 loss: 347.61044681817293
epoch: 580 loss: 348.1237423568964
Train accuracy of the network on the 100000 diagrams: 91.50800 %
Validation accuracy of the network on the 100000 diagrams: 91.40700 %
epoch: 581 loss: 507.5408746674657
epoch: 582 loss: 407.0488807596266
epoch: 583 loss: 352.70354234427214
epoch: 584 loss: 345.92962873354554
epoch: 585 loss: 442.1893859356642
epoch: 586 loss: 358.3528976812959
epoch: 587 loss: 651.2685287743807
epoch: 588 loss: 446.64099975675344
epoch: 589 loss: 397.4348632097244
epoch: 590 loss: 360.79970391094685
Train accuracy of the network on the 100000 diagrams: 91.08400 %
Validation accuracy of the network on the 100000 diagrams: 91.08500 %
epoch: 591 loss: 353.654155112803
epoch: 592 loss: 359.5578687414527
epoch: 593 loss: 348.4319404885173
epoch: 594 loss: 347.33168560266495
epoch: 595 loss: 352.2947546057403
epoch: 596 loss: 346.3563590310514
epoch: 597 loss: 347.3960002101958
epoch: 598 loss: 781.8765392526984
epoch: 599 loss: 437.3800927773118
epoch: 600 loss: 386.56534744799137
Train accuracy of the network on the 100000 diagrams: 90.49900 %
Validation accuracy of the network on the 100000 diagrams: 90.46400 %
epoch: 601 loss: 385.89729161560535
epoch: 602 loss: 366.70153161138296
epoch: 603 loss: 392.6670149564743
epoch: 604 loss: 356.3745708949864
epoch: 605 loss: 352.57022719830275
epoch: 606 loss: 350.3547374457121
epoch: 607 loss: 358.6381221935153
epoch: 608 loss: 352.46607306972146
epoch: 609 loss: 343.1632318980992
epoch: 610 loss: 347.03293070197105
Train accuracy of the network on the 100000 diagrams: 90.94200 %
Validation accuracy of the network on the 100000 diagrams: 90.84100 %
epoch: 611 loss: 351.6728454604745
epoch: 612 loss: 349.25425327569246
epoch: 613 loss: 447.3823623470962
epoch: 614 loss: 346.5845496021211
epoch: 615 loss: 345.3615050762892
epoch: 616 loss: 360.16786527633667
epoch: 617 loss: 371.4233620688319
epoch: 618 loss: 343.5613112822175
epoch: 619 loss: 343.6256439499557
epoch: 620 loss: 344.6495670080185
Train accuracy of the network on the 100000 diagrams: 91.38400 %
Validation accuracy of the network on the 100000 diagrams: 91.25100 %
epoch: 621 loss: 345.83934935182333
epoch: 622 loss: 350.1217064037919
epoch: 623 loss: 341.64851719886065
epoch: 624 loss: 344.263899050653
epoch: 625 loss: 342.77249821648
epoch: 626 loss: 346.5515254922211
epoch: 627 loss: 342.7961123883724
epoch: 628 loss: 345.73362017050385
epoch: 629 loss: 350.4417582638562
epoch: 630 loss: 342.5378273539245
Train accuracy of the network on the 100000 diagrams: 91.46600 %
Validation accuracy of the network on the 100000 diagrams: 91.38800 %
epoch: 631 loss: 340.35856983438134
epoch: 632 loss: 341.0260775499046
epoch: 633 loss: 341.9767857529223
epoch: 634 loss: 354.0535048395395
epoch: 635 loss: 340.4753163829446
epoch: 636 loss: 338.8649633899331
epoch: 637 loss: 360.07731445133686
epoch: 638 loss: 339.05878561735153
epoch: 639 loss: 340.81063863262534
epoch: 640 loss: 338.57555239647627
Train accuracy of the network on the 100000 diagrams: 91.01500 %
Validation accuracy of the network on the 100000 diagrams: 90.90500 %
epoch: 641 loss: 340.6905643604696
epoch: 642 loss: 339.0665641836822
epoch: 643 loss: 340.0916297249496
epoch: 644 loss: 339.6063403226435
epoch: 645 loss: 339.1691118068993
epoch: 646 loss: 339.1313866302371
epoch: 647 loss: 341.0868734009564
epoch: 648 loss: 339.07369024306536
epoch: 649 loss: 381.68227603286505
epoch: 650 loss: 339.7671255208552
Train accuracy of the network on the 100000 diagrams: 91.11600 %
Validation accuracy of the network on the 100000 diagrams: 91.04200 %
epoch: 651 loss: 345.055531386286
epoch: 652 loss: 346.4024547562003
epoch: 653 loss: 337.35567877814174
epoch: 654 loss: 337.86726378649473
epoch: 655 loss: 337.1732557117939
epoch: 656 loss: 340.7056166343391
epoch: 657 loss: 339.4736412316561
epoch: 658 loss: 340.17614877969027
epoch: 659 loss: 339.3916390016675
epoch: 660 loss: 343.7007592022419
Train accuracy of the network on the 100000 diagrams: 91.60100 %
Validation accuracy of the network on the 100000 diagrams: 91.43300 %
epoch: 661 loss: 346.54947348311543
epoch: 662 loss: 338.63958451524377
epoch: 663 loss: 340.50739316642284
epoch: 664 loss: 335.944787055254
epoch: 665 loss: 336.17174491658807
epoch: 666 loss: 351.85640336573124
epoch: 667 loss: 339.7324602045119
epoch: 668 loss: 339.3269016444683
epoch: 669 loss: 336.92005056887865
epoch: 670 loss: 339.1797694452107
Train accuracy of the network on the 100000 diagrams: 91.55000 %
Validation accuracy of the network on the 100000 diagrams: 91.54200 %
epoch: 671 loss: 337.82737235352397
epoch: 672 loss: 341.8485076986253
epoch: 673 loss: 336.1108285486698
epoch: 674 loss: 337.5588403120637
epoch: 675 loss: 339.35187819600105
epoch: 676 loss: 337.6268530189991
epoch: 677 loss: 338.99791158363223
epoch: 678 loss: 339.60530088096857
epoch: 679 loss: 362.1703179180622
epoch: 680 loss: 341.9183818474412
Train accuracy of the network on the 100000 diagrams: 91.17700 %
Validation accuracy of the network on the 100000 diagrams: 91.11800 %
epoch: 681 loss: 339.0925821363926
epoch: 682 loss: 337.1163065135479
epoch: 683 loss: 338.3462912328541
epoch: 684 loss: 338.4662517309189
epoch: 685 loss: 336.7527146600187
epoch: 686 loss: 341.6712118834257
epoch: 687 loss: 336.68243059515953
epoch: 688 loss: 334.3376678004861
epoch: 689 loss: 338.164743706584
epoch: 690 loss: 337.0159343034029
Train accuracy of the network on the 100000 diagrams: 91.39500 %
Validation accuracy of the network on the 100000 diagrams: 91.30000 %
epoch: 691 loss: 337.4313051700592
epoch: 692 loss: 335.69244503974915
epoch: 693 loss: 342.95026649162173
epoch: 694 loss: 335.5082449093461
epoch: 695 loss: 334.8762550726533
epoch: 696 loss: 335.9251636043191
epoch: 697 loss: 335.72107734158635
epoch: 698 loss: 349.62903794646263
epoch: 699 loss: 331.4062146432698
epoch: 700 loss: 336.25531540066004
Train accuracy of the network on the 100000 diagrams: 91.51100 %
Validation accuracy of the network on the 100000 diagrams: 91.38700 %
epoch: 701 loss: 333.9672130532563
epoch: 702 loss: 336.9603938125074
epoch: 703 loss: 337.5077304765582
epoch: 704 loss: 332.53109580278397
epoch: 705 loss: 344.8925876878202
epoch: 706 loss: 334.757701843977
epoch: 707 loss: 361.8305843435228
epoch: 708 loss: 340.950261618942
epoch: 709 loss: 336.61619701609015
epoch: 710 loss: 332.5179176069796
Train accuracy of the network on the 100000 diagrams: 91.67600 %
Validation accuracy of the network on the 100000 diagrams: 91.49800 %
epoch: 711 loss: 346.27823478356004
epoch: 712 loss: 342.49735597893596
epoch: 713 loss: 335.39172715321183
epoch: 714 loss: 334.8358069509268
epoch: 715 loss: 342.9438066035509
epoch: 716 loss: 334.11641074344516
epoch: 717 loss: 335.027013592422
epoch: 718 loss: 334.0192903317511
epoch: 719 loss: 350.6643161699176
epoch: 720 loss: 336.95130371674895
Train accuracy of the network on the 100000 diagrams: 91.55000 %
Validation accuracy of the network on the 100000 diagrams: 91.40000 %
epoch: 721 loss: 332.75270484015346
epoch: 722 loss: 332.9325544759631
epoch: 723 loss: 338.84000638499856
epoch: 724 loss: 337.64518555626273
epoch: 725 loss: 334.6427561286837
epoch: 726 loss: 335.5142377279699
epoch: 727 loss: 469.2789217159152
epoch: 728 loss: 1046.6804331094027
epoch: 729 loss: 471.2665062993765
epoch: 730 loss: 437.1604651212692
Train accuracy of the network on the 100000 diagrams: 89.50200 %
Validation accuracy of the network on the 100000 diagrams: 89.69500 %
epoch: 731 loss: 414.9438768103719
epoch: 732 loss: 402.53187185525894
epoch: 733 loss: 383.4831971824169
epoch: 734 loss: 355.3749732375145
epoch: 735 loss: 341.5014527775347
epoch: 736 loss: 340.2929280884564
epoch: 737 loss: 339.05150597542524
epoch: 738 loss: 337.41999807581306
epoch: 739 loss: 336.5611748211086
epoch: 740 loss: 341.1707337126136
Train accuracy of the network on the 100000 diagrams: 91.53800 %
Validation accuracy of the network on the 100000 diagrams: 91.45300 %
epoch: 741 loss: 336.63517424836755
epoch: 742 loss: 358.7766681872308
epoch: 743 loss: 348.80239559710026
epoch: 744 loss: 338.1269068568945
epoch: 745 loss: 333.76381945610046
epoch: 746 loss: 338.2933904379606
epoch: 747 loss: 364.90440924465656
epoch: 748 loss: 334.92604303732514
epoch: 749 loss: 334.18814942240715
epoch: 750 loss: 337.26499047130346
Train accuracy of the network on the 100000 diagrams: 91.67100 %
Validation accuracy of the network on the 100000 diagrams: 91.55300 %
epoch: 751 loss: 351.11123774573207
epoch: 752 loss: 335.9258057847619
epoch: 753 loss: 333.94403798878193
epoch: 754 loss: 339.9213178753853
epoch: 755 loss: 332.49842166900635
epoch: 756 loss: 336.88175194710493
epoch: 757 loss: 336.89398311823606
epoch: 758 loss: 358.2944623455405
epoch: 759 loss: 333.46548204496503
epoch: 760 loss: 338.435023419559
Train accuracy of the network on the 100000 diagrams: 91.47200 %
Validation accuracy of the network on the 100000 diagrams: 91.31700 %
epoch: 761 loss: 347.9621075205505
epoch: 762 loss: 334.79949597269297
epoch: 763 loss: 334.75585259869695
epoch: 764 loss: 337.3554029017687
epoch: 765 loss: 337.65177883580327
epoch: 766 loss: 331.0484528541565
epoch: 767 loss: 344.0556479841471
epoch: 768 loss: 339.0156438201666
epoch: 769 loss: 346.9377857297659
epoch: 770 loss: 330.73088258877397
Train accuracy of the network on the 100000 diagrams: 91.25100 %
Validation accuracy of the network on the 100000 diagrams: 91.15800 %
epoch: 771 loss: 333.3021418787539
epoch: 772 loss: 331.94609559699893
epoch: 773 loss: 334.3795415684581
epoch: 774 loss: 333.4332385659218
epoch: 775 loss: 334.15620917081833
epoch: 776 loss: 335.5320139043033
epoch: 777 loss: 333.23135352134705
epoch: 778 loss: 334.4264934659004
epoch: 779 loss: 371.5517169907689
epoch: 780 loss: 328.96642392873764
Train accuracy of the network on the 100000 diagrams: 91.51600 %
Validation accuracy of the network on the 100000 diagrams: 91.32500 %
epoch: 781 loss: 331.3440040908754
epoch: 782 loss: 334.59383661299944
epoch: 783 loss: 332.61153246089816
epoch: 784 loss: 331.608685426414
epoch: 785 loss: 333.87148700654507
epoch: 786 loss: 333.7075523287058
epoch: 787 loss: 340.70154502242804
epoch: 788 loss: 333.18922179192305
epoch: 789 loss: 398.43377950787544
epoch: 790 loss: 330.5130257755518
Train accuracy of the network on the 100000 diagrams: 91.67300 %
Validation accuracy of the network on the 100000 diagrams: 91.53700 %
epoch: 791 loss: 332.2586533129215
epoch: 792 loss: 330.5028945170343
epoch: 793 loss: 330.33613531291485
epoch: 794 loss: 347.0296077542007
epoch: 795 loss: 354.8222300335765
epoch: 796 loss: 350.42177948355675
epoch: 797 loss: 341.8454283438623
epoch: 798 loss: 353.8404077515006
epoch: 799 loss: 333.0336109921336
epoch: 800 loss: 332.61991116032004
Train accuracy of the network on the 100000 diagrams: 90.95700 %
Validation accuracy of the network on the 100000 diagrams: 90.85900 %
epoch: 801 loss: 339.81157928332686
epoch: 802 loss: 334.6959855481982
epoch: 803 loss: 333.7720478512347
epoch: 804 loss: 335.6513618975878
epoch: 805 loss: 332.1994510740042
epoch: 806 loss: 334.76867655292153
epoch: 807 loss: 342.47394721210003
epoch: 808 loss: 330.5311969034374
epoch: 809 loss: 334.7934622615576
epoch: 810 loss: 333.05634754151106
Train accuracy of the network on the 100000 diagrams: 91.18000 %
Validation accuracy of the network on the 100000 diagrams: 90.98300 %
epoch: 811 loss: 331.4274665005505
epoch: 812 loss: 334.58041397482157
epoch: 813 loss: 333.7607017196715
epoch: 814 loss: 353.9835666641593
epoch: 815 loss: 330.1686258725822
epoch: 816 loss: 331.015364009887
epoch: 817 loss: 331.4779452085495
epoch: 818 loss: 329.32325134798884
epoch: 819 loss: 332.0574534460902
epoch: 820 loss: 332.80515029653907
Train accuracy of the network on the 100000 diagrams: 90.80700 %
Validation accuracy of the network on the 100000 diagrams: 90.62700 %
epoch: 821 loss: 331.8230973482132
epoch: 822 loss: 344.8585175536573
epoch: 823 loss: 328.47977992519736
epoch: 824 loss: 330.39467848837376
epoch: 825 loss: 329.9304429590702
epoch: 826 loss: 332.4567002430558
epoch: 827 loss: 330.5164663158357
epoch: 828 loss: 331.97732576727867
epoch: 829 loss: 334.7581306286156
epoch: 830 loss: 332.43281039223075
Train accuracy of the network on the 100000 diagrams: 91.67600 %
Validation accuracy of the network on the 100000 diagrams: 91.39700 %
epoch: 831 loss: 334.057584002614
epoch: 832 loss: 334.6750580854714
epoch: 833 loss: 333.8962648883462
epoch: 834 loss: 332.0920494310558
epoch: 835 loss: 329.13157496601343
epoch: 836 loss: 339.6814025901258
epoch: 837 loss: 334.5910828076303
epoch: 838 loss: 331.1376525275409
epoch: 839 loss: 332.74481047317386
epoch: 840 loss: 330.2216268591583
Train accuracy of the network on the 100000 diagrams: 91.78600 %
Validation accuracy of the network on the 100000 diagrams: 91.66000 %
epoch: 841 loss: 330.99934582039714
epoch: 842 loss: 493.381196834147
epoch: 843 loss: 498.0988331437111
epoch: 844 loss: 410.71259012818336
epoch: 845 loss: 360.6072117462754
epoch: 846 loss: 328.8974938504398
epoch: 847 loss: 390.1731571853161
epoch: 848 loss: 341.3002978041768
epoch: 849 loss: 581.2508098371327
epoch: 850 loss: 366.2119464799762
Train accuracy of the network on the 100000 diagrams: 91.49400 %
Validation accuracy of the network on the 100000 diagrams: 91.40100 %
epoch: 851 loss: 344.93804470449686
epoch: 852 loss: 344.7915995121002
epoch: 853 loss: 338.0441854298115
epoch: 854 loss: 335.77756586670876
epoch: 855 loss: 335.8270768523216
epoch: 856 loss: 433.0316591113806
epoch: 857 loss: 336.3223814368248
epoch: 858 loss: 328.6885082870722
epoch: 859 loss: 329.70207135006785
epoch: 860 loss: 391.93349350988865
Train accuracy of the network on the 100000 diagrams: 91.21000 %
Validation accuracy of the network on the 100000 diagrams: 91.08900 %
epoch: 861 loss: 333.9559488967061
epoch: 862 loss: 350.0517603009939
epoch: 863 loss: 858.9740672260523
epoch: 864 loss: 425.4306475818157
epoch: 865 loss: 408.8262269794941
epoch: 866 loss: 386.9226178377867
epoch: 867 loss: 377.78717214614153
epoch: 868 loss: 374.98346092551947
epoch: 869 loss: 379.010088019073
epoch: 870 loss: 363.43945167958736
Train accuracy of the network on the 100000 diagrams: 90.85100 %
Validation accuracy of the network on the 100000 diagrams: 90.78500 %
epoch: 871 loss: 365.5590369142592
epoch: 872 loss: 357.8987884186208
epoch: 873 loss: 349.47572557255626
epoch: 874 loss: 352.4337223023176
epoch: 875 loss: 348.16443422436714
epoch: 876 loss: 348.981060102582
epoch: 877 loss: 341.4894374459982
epoch: 878 loss: 342.08222349360585
epoch: 879 loss: 347.26841992139816
epoch: 880 loss: 340.9866959378123
Train accuracy of the network on the 100000 diagrams: 91.44700 %
Validation accuracy of the network on the 100000 diagrams: 91.43000 %
epoch: 881 loss: 339.48130448162556
epoch: 882 loss: 346.51217368245125
epoch: 883 loss: 337.7554753795266
epoch: 884 loss: 357.1482150480151
epoch: 885 loss: 864.7891281917691
epoch: 886 loss: 676.94148260355
epoch: 887 loss: 595.0375143587589
epoch: 888 loss: 488.6390671133995
epoch: 889 loss: 416.50765538588166
epoch: 890 loss: 399.2419700548053
Train accuracy of the network on the 100000 diagrams: 89.91700 %
Validation accuracy of the network on the 100000 diagrams: 90.01200 %
epoch: 891 loss: 381.81335147470236
epoch: 892 loss: 665.1567262858152
epoch: 893 loss: 692.4665451869369
epoch: 894 loss: 478.87951248139143
epoch: 895 loss: 465.0370720922947
epoch: 896 loss: 465.1742971614003
epoch: 897 loss: 478.98500968515873
epoch: 898 loss: 450.23918841034174
epoch: 899 loss: 489.05184815078974
epoch: 900 loss: 475.32392022013664
Train accuracy of the network on the 100000 diagrams: 90.28000 %
Validation accuracy of the network on the 100000 diagrams: 90.39200 %
epoch: 901 loss: 397.56472251564264
epoch: 902 loss: 392.80079490691423
epoch: 903 loss: 390.1598157733679
epoch: 904 loss: 423.61126974225044
epoch: 905 loss: 400.63165184110403
epoch: 906 loss: 380.1962511241436
epoch: 907 loss: 376.0444427654147
epoch: 908 loss: 493.65047857165337
epoch: 909 loss: 418.4659680351615
epoch: 910 loss: 375.09295934438705
Train accuracy of the network on the 100000 diagrams: 90.57500 %
Validation accuracy of the network on the 100000 diagrams: 90.54800 %
epoch: 911 loss: 422.05357226729393
epoch: 912 loss: 370.2689509987831
epoch: 913 loss: 374.19325095415115
epoch: 914 loss: 381.820977114141
epoch: 915 loss: 366.9999717026949
epoch: 916 loss: 366.844518430531
epoch: 917 loss: 362.53139359131455
epoch: 918 loss: 361.46383499354124
epoch: 919 loss: 374.59455785900354
epoch: 920 loss: 362.2731111124158
Train accuracy of the network on the 100000 diagrams: 91.21800 %
Validation accuracy of the network on the 100000 diagrams: 91.26500 %
epoch: 921 loss: 358.6559020727873
epoch: 922 loss: 351.2312030121684
epoch: 923 loss: 365.6739168204367
epoch: 924 loss: 361.222599953413
epoch: 925 loss: 366.3726156204939
epoch: 926 loss: 353.0273449085653
epoch: 927 loss: 351.37438639998436
epoch: 928 loss: 359.93435348570347
epoch: 929 loss: 347.6151359975338
epoch: 930 loss: 354.2016711086035
Train accuracy of the network on the 100000 diagrams: 91.32700 %
Validation accuracy of the network on the 100000 diagrams: 91.27800 %
epoch: 931 loss: 352.28835068270564
epoch: 932 loss: 347.8620340898633
epoch: 933 loss: 347.73257503658533
epoch: 934 loss: 353.88150716573
epoch: 935 loss: 347.23525481298566
epoch: 936 loss: 352.35666443780065
epoch: 937 loss: 344.6662088893354
epoch: 938 loss: 345.7854522578418
epoch: 939 loss: 344.6322246789932
epoch: 940 loss: 345.2806367725134
Train accuracy of the network on the 100000 diagrams: 91.41700 %
Validation accuracy of the network on the 100000 diagrams: 91.34100 %
epoch: 941 loss: 344.71383422613144
epoch: 942 loss: 343.11707158759236
epoch: 943 loss: 341.3688355255872
epoch: 944 loss: 351.8921017125249
epoch: 945 loss: 343.6117171347141
epoch: 946 loss: 338.9980556406081
epoch: 947 loss: 341.5923124887049
epoch: 948 loss: 802.3744468353689
epoch: 949 loss: 446.8221201002598
epoch: 950 loss: 378.828620031476
Train accuracy of the network on the 100000 diagrams: 90.92200 %
Validation accuracy of the network on the 100000 diagrams: 90.97900 %
epoch: 951 loss: 352.473452873528
epoch: 952 loss: 364.6339848674834
epoch: 953 loss: 374.3945477306843
epoch: 954 loss: 342.8287390433252
epoch: 955 loss: 344.52657952159643
epoch: 956 loss: 345.9429859034717
epoch: 957 loss: 526.5583065673709
epoch: 958 loss: 343.0561823360622
epoch: 959 loss: 338.73520408570766
epoch: 960 loss: 427.7659136541188
Train accuracy of the network on the 100000 diagrams: 91.26400 %
Validation accuracy of the network on the 100000 diagrams: 91.22600 %
epoch: 961 loss: 393.31646224856377
epoch: 962 loss: 750.1155004650354
epoch: 963 loss: 457.7789409980178
epoch: 964 loss: 420.41583455353975
epoch: 965 loss: 413.1037010252476
epoch: 966 loss: 519.4246591031551
epoch: 967 loss: 409.33364333212376
epoch: 968 loss: 405.069876357913
epoch: 969 loss: 396.45314835757017
epoch: 970 loss: 389.7535487487912
Train accuracy of the network on the 100000 diagrams: 90.07100 %
Validation accuracy of the network on the 100000 diagrams: 90.12700 %
epoch: 971 loss: 378.8440552651882
epoch: 972 loss: 374.92868061363697
epoch: 973 loss: 373.6356118991971
epoch: 974 loss: 362.15924602374434
epoch: 975 loss: 404.76137605309486
epoch: 976 loss: 368.9699682146311
epoch: 977 loss: 361.0528370887041
epoch: 978 loss: 410.3413551785052
epoch: 979 loss: 367.1839169487357
epoch: 980 loss: 380.18579491227865
Train accuracy of the network on the 100000 diagrams: 90.99500 %
Validation accuracy of the network on the 100000 diagrams: 91.00100 %
epoch: 981 loss: 355.5721269771457
epoch: 982 loss: 359.98302383720875
epoch: 983 loss: 470.5930416584015
epoch: 984 loss: 375.3726850040257
epoch: 985 loss: 383.2576719112694
epoch: 986 loss: 360.4267498701811
epoch: 987 loss: 364.2080933377147
epoch: 988 loss: 368.4185364767909
epoch: 989 loss: 351.82383140176535
epoch: 990 loss: 356.9928832910955
Train accuracy of the network on the 100000 diagrams: 91.34500 %
Validation accuracy of the network on the 100000 diagrams: 91.28000 %
epoch: 991 loss: 351.0847516916692
epoch: 992 loss: 362.7189164273441
epoch: 993 loss: 351.24626832827926
epoch: 994 loss: 347.13109981268644
epoch: 995 loss: 347.89349932968616
epoch: 996 loss: 377.54177098348737
epoch: 997 loss: 348.30876626074314
epoch: 998 loss: 526.8645470961928
epoch: 999 loss: 416.0593976229429
epoch: 0 loss: 402.4503660649061
Train accuracy of the network on the 100000 diagrams: 90.96400 %
Validation accuracy of the network on the 100000 diagrams: 90.99500 %
epoch: 1 loss: 353.86164227873087
epoch: 2 loss: 347.1597812175751
epoch: 3 loss: 345.3440239429474
epoch: 4 loss: 348.17922411859035
epoch: 5 loss: 423.6555027663708
epoch: 6 loss: 361.6676327139139
epoch: 7 loss: 343.8860522136092
epoch: 8 loss: 348.2933304645121
epoch: 9 loss: 347.2049731053412
epoch: 10 loss: 345.10646193102
Train accuracy of the network on the 100000 diagrams: 91.18800 %
Validation accuracy of the network on the 100000 diagrams: 91.09200 %
epoch: 11 loss: 377.53261613845825
epoch: 12 loss: 346.3925321996212
epoch: 13 loss: 342.63630766421556
epoch: 14 loss: 343.15135192126036
epoch: 15 loss: 343.29575265198946
epoch: 16 loss: 350.3774123713374
epoch: 17 loss: 344.26803678646684
epoch: 18 loss: 343.0398129671812
epoch: 19 loss: 345.3840432912111
epoch: 20 loss: 347.1961388029158
Train accuracy of the network on the 100000 diagrams: 91.17500 %
Validation accuracy of the network on the 100000 diagrams: 91.21000 %
epoch: 21 loss: 340.3027379065752
epoch: 22 loss: 344.6495099067688
epoch: 23 loss: 346.8989525362849
epoch: 24 loss: 339.0545823350549
epoch: 25 loss: 341.0443252399564
epoch: 26 loss: 340.31058334186673
epoch: 27 loss: 339.75739577412605
epoch: 28 loss: 338.8854636698961
epoch: 29 loss: 338.6074110902846
epoch: 30 loss: 339.4743455834687
Train accuracy of the network on the 100000 diagrams: 91.40100 %
Validation accuracy of the network on the 100000 diagrams: 91.30700 %
epoch: 31 loss: 340.63156444206834
epoch: 32 loss: 405.241339687258
epoch: 33 loss: 343.2389232069254
epoch: 34 loss: 338.6885714940727
epoch: 35 loss: 336.15249759703875
epoch: 36 loss: 339.90997130423784
epoch: 37 loss: 341.3705925308168
epoch: 38 loss: 339.19448363780975
epoch: 39 loss: 343.019301969558
epoch: 40 loss: 340.44827807694674
Train accuracy of the network on the 100000 diagrams: 91.52500 %
Validation accuracy of the network on the 100000 diagrams: 91.29900 %
epoch: 41 loss: 335.7879730090499
epoch: 42 loss: 341.57193790376186
epoch: 43 loss: 345.8503894433379
epoch: 44 loss: 336.14727079868317
epoch: 45 loss: 340.1963883712888
epoch: 46 loss: 337.69650937616825
epoch: 47 loss: 338.3835226781666
epoch: 48 loss: 336.6854013390839
epoch: 49 loss: 336.56642666086555
epoch: 50 loss: 338.0805357694626
Train accuracy of the network on the 100000 diagrams: 91.67600 %
Validation accuracy of the network on the 100000 diagrams: 91.54900 %
epoch: 51 loss: 337.58629858866334
epoch: 52 loss: 336.59769601747394
epoch: 53 loss: 341.0783778652549
epoch: 54 loss: 338.5831560753286
epoch: 55 loss: 334.85971263051033
epoch: 56 loss: 345.543843280524
epoch: 57 loss: 367.0017254911363
epoch: 58 loss: 334.9957135133445
epoch: 59 loss: 333.30556666478515
epoch: 60 loss: 345.1202913187444
Train accuracy of the network on the 100000 diagrams: 91.58900 %
Validation accuracy of the network on the 100000 diagrams: 91.50300 %
epoch: 61 loss: 332.7885714918375
epoch: 62 loss: 334.8932988643646
epoch: 63 loss: 335.3842943832278
epoch: 64 loss: 410.4956334158778
epoch: 65 loss: 337.9298703186214
epoch: 66 loss: 334.36480345577
epoch: 67 loss: 339.39518202468753
epoch: 68 loss: 337.2163933813572
epoch: 69 loss: 338.09101262688637
epoch: 70 loss: 335.3458950147033
Train accuracy of the network on the 100000 diagrams: 90.70600 %
Validation accuracy of the network on the 100000 diagrams: 90.60200 %
epoch: 71 loss: 381.7686577923596
epoch: 72 loss: 334.0551058910787
epoch: 73 loss: 330.93051686510444
epoch: 74 loss: 332.2380553558469
epoch: 75 loss: 334.8918962776661
epoch: 76 loss: 335.95365254580975
epoch: 77 loss: 332.71022913977504
epoch: 78 loss: 333.89816040173173
epoch: 79 loss: 348.61603163555264
epoch: 80 loss: 336.7953389659524
Train accuracy of the network on the 100000 diagrams: 91.55100 %
Validation accuracy of the network on the 100000 diagrams: 91.33100 %
epoch: 81 loss: 332.53046219423413
epoch: 82 loss: 334.9413762614131
epoch: 83 loss: 340.8605098053813
epoch: 84 loss: 337.35679922252893
epoch: 85 loss: 333.895032890141
epoch: 86 loss: 333.8911740258336
epoch: 87 loss: 334.1131258904934
epoch: 88 loss: 339.05910800769925
epoch: 89 loss: 365.6993447057903
epoch: 90 loss: 335.79823242127895
Train accuracy of the network on the 100000 diagrams: 91.52200 %
Validation accuracy of the network on the 100000 diagrams: 91.29900 %
epoch: 91 loss: 331.6186662763357
epoch: 92 loss: 334.7933209091425
epoch: 93 loss: 396.79117422923446
epoch: 94 loss: 331.1988035365939
epoch: 95 loss: 333.2599974051118
epoch: 96 loss: 345.4861630499363
epoch: 97 loss: 335.11043022200465
epoch: 98 loss: 336.26914051547647
epoch: 99 loss: 332.887427020818
epoch: 100 loss: 334.76251359656453
Train accuracy of the network on the 100000 diagrams: 91.49800 %
Validation accuracy of the network on the 100000 diagrams: 91.38900 %
epoch: 101 loss: 333.859729681164
epoch: 102 loss: 334.301218777895
epoch: 103 loss: 337.8470040857792
epoch: 104 loss: 332.894528247416
epoch: 105 loss: 331.5854356698692
epoch: 106 loss: 334.5011106990278
epoch: 107 loss: 334.0327130779624
epoch: 108 loss: 332.3407514989376
epoch: 109 loss: 334.2256332933903
epoch: 110 loss: 340.44816318154335
Train accuracy of the network on the 100000 diagrams: 90.87400 %
Validation accuracy of the network on the 100000 diagrams: 90.82000 %
epoch: 111 loss: 762.576968614012
epoch: 112 loss: 343.87361029163003
epoch: 113 loss: 350.0302103534341
epoch: 114 loss: 351.1769187487662
epoch: 115 loss: 396.9896970503032
epoch: 116 loss: 347.81077483668923
epoch: 117 loss: 349.51715157181025
epoch: 118 loss: 332.4689651913941
epoch: 119 loss: 335.6627382338047
epoch: 120 loss: 338.03759257122874
Train accuracy of the network on the 100000 diagrams: 91.36600 %
Validation accuracy of the network on the 100000 diagrams: 91.18100 %
epoch: 121 loss: 352.13393998518586
epoch: 122 loss: 337.3962921537459
epoch: 123 loss: 335.740088339895
epoch: 124 loss: 387.3710391148925
epoch: 125 loss: 493.6246663108468
epoch: 126 loss: 369.40845028683543
epoch: 127 loss: 357.93478313088417
epoch: 128 loss: 369.5863092467189
epoch: 129 loss: 349.74895268678665
epoch: 130 loss: 397.55377159267664
Train accuracy of the network on the 100000 diagrams: 91.23700 %
Validation accuracy of the network on the 100000 diagrams: 90.98500 %
epoch: 131 loss: 3630.603723678738
epoch: 132 loss: 968.4174503087997
epoch: 133 loss: 846.7619929611683
epoch: 134 loss: 791.4372880160809
epoch: 135 loss: 641.6333368718624
epoch: 136 loss: 613.9101492464542
epoch: 137 loss: 535.7150144577026
epoch: 138 loss: 491.3675029873848
epoch: 139 loss: 466.54682862758636
epoch: 140 loss: 457.3399602547288
Train accuracy of the network on the 100000 diagrams: 88.80000 %
Validation accuracy of the network on the 100000 diagrams: 88.85500 %
epoch: 141 loss: 448.1042624115944
epoch: 142 loss: 422.89001692831516
epoch: 143 loss: 408.6043684259057
epoch: 144 loss: 482.7718395218253
epoch: 145 loss: 435.36500485241413
epoch: 146 loss: 402.04599694535136
epoch: 147 loss: 408.8405603989959
epoch: 148 loss: 388.02168000489473
epoch: 149 loss: 391.8456016778946
epoch: 150 loss: 391.17128271609545
Train accuracy of the network on the 100000 diagrams: 90.68600 %
Validation accuracy of the network on the 100000 diagrams: 90.70800 %
epoch: 151 loss: 389.22164261341095
epoch: 152 loss: 383.68976103514433
epoch: 153 loss: 385.6807275041938
epoch: 154 loss: 378.73567274957895
epoch: 155 loss: 390.84482222795486
epoch: 156 loss: 397.48131200671196
epoch: 157 loss: 379.6562074199319
epoch: 158 loss: 373.4210508465767
epoch: 159 loss: 371.38925410062075
epoch: 160 loss: 373.1662958599627
Train accuracy of the network on the 100000 diagrams: 90.12900 %
Validation accuracy of the network on the 100000 diagrams: 90.10500 %
epoch: 161 loss: 374.1447521299124
epoch: 162 loss: 375.84094070643187
epoch: 163 loss: 377.36832881718874
epoch: 164 loss: 364.72325222939253
epoch: 165 loss: 369.3569099754095
epoch: 166 loss: 365.70775463059545
epoch: 167 loss: 364.5524104461074
epoch: 168 loss: 365.6136959269643
epoch: 169 loss: 366.4232092052698
epoch: 170 loss: 370.2736425884068
Train accuracy of the network on the 100000 diagrams: 90.78400 %
Validation accuracy of the network on the 100000 diagrams: 90.85500 %
epoch: 171 loss: 367.43134830892086
epoch: 172 loss: 360.7817210108042
epoch: 173 loss: 357.90577932074666
epoch: 174 loss: 369.4054308831692
epoch: 175 loss: 360.941106017679
epoch: 176 loss: 361.5962305739522
epoch: 177 loss: 356.7477268688381
epoch: 178 loss: 358.6893557347357
epoch: 179 loss: 356.4560586884618
epoch: 180 loss: 356.558701954782
Train accuracy of the network on the 100000 diagrams: 91.14800 %
Validation accuracy of the network on the 100000 diagrams: 91.14500 %
epoch: 181 loss: 357.3578310497105
epoch: 182 loss: 354.97136425971985
epoch: 183 loss: 388.6045189499855
epoch: 184 loss: 360.5996421724558
epoch: 185 loss: 354.21671030670404
epoch: 186 loss: 357.27244760096073
epoch: 187 loss: 354.297376960516
epoch: 188 loss: 360.7415739670396
epoch: 189 loss: 354.7004166841507
epoch: 190 loss: 358.1947643458843
Train accuracy of the network on the 100000 diagrams: 90.31700 %
Validation accuracy of the network on the 100000 diagrams: 90.36900 %
epoch: 191 loss: 353.88660760223866
epoch: 192 loss: 351.8040142431855
epoch: 193 loss: 353.52761025726795
epoch: 194 loss: 357.27381309866905
epoch: 195 loss: 353.54261131584644
epoch: 196 loss: 349.65394434705377
epoch: 197 loss: 351.72386882454157
epoch: 198 loss: 349.1569554358721
epoch: 199 loss: 352.1775233447552
epoch: 200 loss: 365.2045837044716
Train accuracy of the network on the 100000 diagrams: 90.68800 %
Validation accuracy of the network on the 100000 diagrams: 90.77300 %
epoch: 201 loss: 358.95220778882504
epoch: 202 loss: 349.6406352519989
epoch: 203 loss: 355.76246961578727
epoch: 204 loss: 349.45330262184143
epoch: 205 loss: 347.63322151452303
epoch: 206 loss: 366.4181561432779
epoch: 207 loss: 355.4791728183627
epoch: 208 loss: 373.84139732271433
epoch: 209 loss: 356.93993151932955
epoch: 210 loss: 354.71487203240395
Train accuracy of the network on the 100000 diagrams: 91.11800 %
Validation accuracy of the network on the 100000 diagrams: 91.03100 %
epoch: 211 loss: 348.4052210301161
epoch: 212 loss: 359.13712380453944
epoch: 213 loss: 351.14200516045094
epoch: 214 loss: 348.0695116147399
epoch: 215 loss: 501.8623707443476
epoch: 216 loss: 379.4989721029997
epoch: 217 loss: 387.4253758005798
epoch: 218 loss: 368.0433285459876
epoch: 219 loss: 384.713390275836
epoch: 220 loss: 370.5037386007607
Train accuracy of the network on the 100000 diagrams: 90.30800 %
Validation accuracy of the network on the 100000 diagrams: 90.26800 %
epoch: 221 loss: 371.5454538241029
epoch: 222 loss: 396.69854302704334
epoch: 223 loss: 384.80405205115676
epoch: 224 loss: 358.46307537704706
epoch: 225 loss: 354.26538180559874
epoch: 226 loss: 355.77652526274323
epoch: 227 loss: 363.52061884850264
epoch: 228 loss: 351.447187744081
epoch: 229 loss: 349.77965665981174
epoch: 230 loss: 349.0903697013855
Train accuracy of the network on the 100000 diagrams: 90.93500 %
Validation accuracy of the network on the 100000 diagrams: 90.88500 %
epoch: 231 loss: 350.7131845355034
epoch: 232 loss: 349.468582354486
epoch: 233 loss: 349.6521070599556
epoch: 234 loss: 349.1354811154306
epoch: 235 loss: 349.9346876293421
epoch: 236 loss: 348.03625575080514
epoch: 237 loss: 347.9314960539341
epoch: 238 loss: 345.89784475043416
epoch: 239 loss: 347.1250877417624
epoch: 240 loss: 347.9536176174879
Train accuracy of the network on the 100000 diagrams: 91.30000 %
Validation accuracy of the network on the 100000 diagrams: 91.22100 %
epoch: 241 loss: 347.3303051441908
epoch: 242 loss: 348.6087360456586
epoch: 243 loss: 346.59801810607314
epoch: 244 loss: 341.5521667711437
epoch: 245 loss: 345.2242255695164
epoch: 246 loss: 350.59952611476183
epoch: 247 loss: 343.1539213992655
epoch: 248 loss: 378.5040065571666
epoch: 249 loss: 347.5713246203959
epoch: 250 loss: 349.2791801430285
Train accuracy of the network on the 100000 diagrams: 91.32000 %
Validation accuracy of the network on the 100000 diagrams: 91.24400 %
epoch: 251 loss: 347.63226494193077
epoch: 252 loss: 342.84213469177485
epoch: 253 loss: 343.03116420283914
epoch: 254 loss: 347.6948511302471
epoch: 255 loss: 345.6003521159291
epoch: 256 loss: 345.7451054006815
epoch: 257 loss: 344.6076454743743
epoch: 258 loss: 343.7448503449559
epoch: 259 loss: 342.5500960871577
epoch: 260 loss: 344.94558349996805
Train accuracy of the network on the 100000 diagrams: 90.95300 %
Validation accuracy of the network on the 100000 diagrams: 90.98900 %
epoch: 261 loss: 345.50358847528696
epoch: 262 loss: 344.01175379380584
epoch: 263 loss: 341.3223140053451
epoch: 264 loss: 342.7215311639011
epoch: 265 loss: 342.3486781604588
epoch: 266 loss: 350.15123695135117
epoch: 267 loss: 338.4965764209628
epoch: 268 loss: 341.43173694983125
epoch: 269 loss: 343.1661099344492
epoch: 270 loss: 344.26899967342615
Train accuracy of the network on the 100000 diagrams: 91.24700 %
Validation accuracy of the network on the 100000 diagrams: 91.13200 %
epoch: 271 loss: 343.2716166526079
epoch: 272 loss: 339.3113008290529
epoch: 273 loss: 343.50661555305123
epoch: 274 loss: 339.4933811686933
epoch: 275 loss: 341.2103068456054
epoch: 276 loss: 355.85998410731554
epoch: 277 loss: 335.71883523464203
epoch: 278 loss: 371.79347512871027
epoch: 279 loss: 338.74459310248494
epoch: 280 loss: 342.58525940030813
Train accuracy of the network on the 100000 diagrams: 91.10500 %
Validation accuracy of the network on the 100000 diagrams: 91.00700 %
epoch: 281 loss: 342.38685703650117
epoch: 282 loss: 342.1662260219455
epoch: 283 loss: 341.90660831704736
epoch: 284 loss: 341.8032526895404
epoch: 285 loss: 340.5717968735844
epoch: 286 loss: 340.17472152784467
epoch: 287 loss: 342.9105735383928
epoch: 288 loss: 399.64431136474013
epoch: 289 loss: 366.1834428906441
epoch: 290 loss: 348.76930325105786
Train accuracy of the network on the 100000 diagrams: 91.03500 %
Validation accuracy of the network on the 100000 diagrams: 90.96900 %
epoch: 291 loss: 348.64596489071846
epoch: 292 loss: 492.47953421249986
epoch: 293 loss: 351.94241689145565
epoch: 294 loss: 347.0651323199272
epoch: 295 loss: 344.14934531599283
epoch: 296 loss: 346.41490334272385
epoch: 297 loss: 342.1889090947807
epoch: 298 loss: 341.86037443578243
epoch: 299 loss: 342.45444625243545
epoch: 300 loss: 342.77703788131475
Train accuracy of the network on the 100000 diagrams: 91.31200 %
Validation accuracy of the network on the 100000 diagrams: 91.32300 %
epoch: 301 loss: 341.45867090672255
epoch: 302 loss: 340.95449931919575
epoch: 303 loss: 338.78852670639753
epoch: 304 loss: 348.74420042335987
epoch: 305 loss: 344.6553194411099
epoch: 306 loss: 338.26080678775907
epoch: 307 loss: 357.9245237931609
epoch: 308 loss: 339.163308352232
epoch: 309 loss: 344.0605418533087
epoch: 310 loss: 347.58089127391577
Train accuracy of the network on the 100000 diagrams: 91.35600 %
Validation accuracy of the network on the 100000 diagrams: 91.39200 %
epoch: 311 loss: 341.0129371844232
epoch: 312 loss: 338.6175794750452
epoch: 313 loss: 340.0360310897231
epoch: 314 loss: 339.08206284046173
epoch: 315 loss: 341.47942232340574
epoch: 316 loss: 339.97588968649507
epoch: 317 loss: 338.9602216631174
epoch: 318 loss: 342.95558400824666
epoch: 319 loss: 338.1333020180464
epoch: 320 loss: 339.88464614748955
Train accuracy of the network on the 100000 diagrams: 91.47500 %
Validation accuracy of the network on the 100000 diagrams: 91.43000 %
epoch: 321 loss: 342.88125382363796
epoch: 322 loss: 340.9901655949652
epoch: 323 loss: 338.08925454318523
epoch: 324 loss: 340.2916442193091
epoch: 325 loss: 337.9775766655803
epoch: 326 loss: 339.65567568689585
epoch: 327 loss: 344.7947137616575
epoch: 328 loss: 337.8596259355545
epoch: 329 loss: 339.19379034638405
epoch: 330 loss: 336.8522941842675
Train accuracy of the network on the 100000 diagrams: 91.40600 %
Validation accuracy of the network on the 100000 diagrams: 91.25700 %
epoch: 331 loss: 338.89772395044565
epoch: 332 loss: 337.7118910215795
epoch: 333 loss: 336.8746953494847
epoch: 334 loss: 340.33688828349113
epoch: 335 loss: 337.51936476677656
epoch: 336 loss: 349.14645193889737
epoch: 337 loss: 334.43880396708846
epoch: 338 loss: 336.4318845048547
epoch: 339 loss: 335.38331523537636
epoch: 340 loss: 345.1408170387149
Train accuracy of the network on the 100000 diagrams: 91.19400 %
Validation accuracy of the network on the 100000 diagrams: 91.10900 %
epoch: 341 loss: 336.718886166811
epoch: 342 loss: 349.4625240787864
epoch: 343 loss: 340.24626717716455
epoch: 344 loss: 345.59772650897503
epoch: 345 loss: 364.74433639645576
epoch: 346 loss: 344.1867413148284
epoch: 347 loss: 338.2623203918338
epoch: 348 loss: 337.55994368717074
epoch: 349 loss: 338.51119738817215
epoch: 350 loss: 337.3507961407304
Train accuracy of the network on the 100000 diagrams: 90.54300 %
Validation accuracy of the network on the 100000 diagrams: 90.53700 %
epoch: 351 loss: 339.87045823410153
epoch: 352 loss: 337.1013779863715
epoch: 353 loss: 338.1362596228719
epoch: 354 loss: 335.2651154734194
epoch: 355 loss: 337.9717122018337
epoch: 356 loss: 338.7992787733674
epoch: 357 loss: 338.6653509736061
epoch: 358 loss: 338.18393353000283
epoch: 359 loss: 334.8599754087627
epoch: 360 loss: 392.76473696529865
Train accuracy of the network on the 100000 diagrams: 82.52600 %
Validation accuracy of the network on the 100000 diagrams: 82.87200 %
epoch: 361 loss: 368.5666874907911
epoch: 362 loss: 339.9479125700891
epoch: 363 loss: 346.68779101222754
epoch: 364 loss: 337.17010724917054
epoch: 365 loss: 336.8725641295314
epoch: 366 loss: 337.46261186152697
epoch: 367 loss: 339.74037166312337
epoch: 368 loss: 335.9217981994152
epoch: 369 loss: 371.38716411590576
epoch: 370 loss: 338.3908794410527
Train accuracy of the network on the 100000 diagrams: 91.05300 %
Validation accuracy of the network on the 100000 diagrams: 90.87400 %
epoch: 371 loss: 335.18486699834466
epoch: 372 loss: 335.0614227876067
epoch: 373 loss: 334.0532891564071
epoch: 374 loss: 335.7408244870603
epoch: 375 loss: 338.8187377601862
epoch: 376 loss: 337.2812083736062
epoch: 377 loss: 335.3407000899315
epoch: 378 loss: 338.64616406708956
epoch: 379 loss: 335.9496951624751
epoch: 380 loss: 335.6489533036947
Train accuracy of the network on the 100000 diagrams: 91.58300 %
Validation accuracy of the network on the 100000 diagrams: 91.38400 %
epoch: 381 loss: 472.42562828958035
epoch: 382 loss: 370.72726552933455
epoch: 383 loss: 341.73284639418125
epoch: 384 loss: 356.16320857033134
epoch: 385 loss: 343.09637951105833
epoch: 386 loss: 337.5758226402104
epoch: 387 loss: 345.7964111380279
epoch: 388 loss: 358.3087556362152
epoch: 389 loss: 368.3245634511113
epoch: 390 loss: 390.78772872686386
Train accuracy of the network on the 100000 diagrams: 90.83900 %
Validation accuracy of the network on the 100000 diagrams: 90.81600 %
epoch: 391 loss: 353.1305801868439
epoch: 392 loss: 349.3315878212452
epoch: 393 loss: 342.78435076773167
epoch: 394 loss: 338.76772152632475
epoch: 395 loss: 344.0810260809958
epoch: 396 loss: 339.55873065814376
epoch: 397 loss: 342.835141826421
epoch: 398 loss: 336.47725067287683
epoch: 399 loss: 352.36936635896564
epoch: 400 loss: 342.29003408178687
Train accuracy of the network on the 100000 diagrams: 91.19700 %
Validation accuracy of the network on the 100000 diagrams: 91.10100 %
epoch: 401 loss: 339.9685516692698
epoch: 402 loss: 336.3862220942974
epoch: 403 loss: 336.5367726162076
epoch: 404 loss: 355.1515573114157
epoch: 405 loss: 337.95415318384767
epoch: 406 loss: 337.85172893479466
epoch: 407 loss: 334.63079683482647
epoch: 408 loss: 336.6936837658286
epoch: 409 loss: 334.49853179231286
epoch: 410 loss: 337.95594089478254
Train accuracy of the network on the 100000 diagrams: 90.74200 %
Validation accuracy of the network on the 100000 diagrams: 90.61700 %
epoch: 411 loss: 341.5620521232486
epoch: 412 loss: 335.3080351315439
epoch: 413 loss: 333.3776529803872
epoch: 414 loss: 336.7456426098943
epoch: 415 loss: 335.5073015578091
epoch: 416 loss: 334.3898262567818
epoch: 417 loss: 335.6775937192142
epoch: 418 loss: 333.57267278432846
epoch: 419 loss: 336.55511516705155
epoch: 420 loss: 334.8837839663029
Train accuracy of the network on the 100000 diagrams: 91.37400 %
Validation accuracy of the network on the 100000 diagrams: 91.33800 %
epoch: 421 loss: 334.03703501448035
epoch: 422 loss: 348.741189327091
epoch: 423 loss: 331.9925778694451
epoch: 424 loss: 331.8443585149944
epoch: 425 loss: 334.1283511891961
epoch: 426 loss: 334.0300283767283
epoch: 427 loss: 337.2101320140064
epoch: 428 loss: 337.1072129867971
epoch: 429 loss: 333.3271885961294
epoch: 430 loss: 342.9813049584627
Train accuracy of the network on the 100000 diagrams: 91.25900 %
Validation accuracy of the network on the 100000 diagrams: 91.14200 %
epoch: 431 loss: 336.13442020863295
epoch: 432 loss: 331.3845168873668
epoch: 433 loss: 332.9382881037891
epoch: 434 loss: 334.89383228495717
epoch: 435 loss: 332.06010134890676
epoch: 436 loss: 331.9857693836093
epoch: 437 loss: 352.3809086084366
epoch: 438 loss: 344.63287184759974
epoch: 439 loss: 350.78203945606947
epoch: 440 loss: 359.88894606754184
Train accuracy of the network on the 100000 diagrams: 91.43700 %
Validation accuracy of the network on the 100000 diagrams: 91.33500 %
epoch: 441 loss: 373.1290323585272
epoch: 442 loss: 333.58462749421597
epoch: 443 loss: 337.72722935676575
epoch: 444 loss: 341.453259434551
epoch: 445 loss: 335.6363639347255
epoch: 446 loss: 374.5393471866846
epoch: 447 loss: 337.54677329957485
epoch: 448 loss: 333.47715232521296
epoch: 449 loss: 336.2729215361178
epoch: 450 loss: 337.5870717316866
Train accuracy of the network on the 100000 diagrams: 91.22900 %
Validation accuracy of the network on the 100000 diagrams: 91.16800 %
epoch: 451 loss: 334.1976844407618
epoch: 452 loss: 335.2502495907247
epoch: 453 loss: 336.2719745486975
epoch: 454 loss: 335.6142864599824
epoch: 455 loss: 334.8377875983715
epoch: 456 loss: 335.9399101808667
epoch: 457 loss: 332.4604598879814
epoch: 458 loss: 330.417817927897
epoch: 459 loss: 341.07825749740005
epoch: 460 loss: 330.5024858750403
Train accuracy of the network on the 100000 diagrams: 91.47300 %
Validation accuracy of the network on the 100000 diagrams: 91.39700 %
epoch: 461 loss: 338.95269994437695
epoch: 462 loss: 330.8535179309547
epoch: 463 loss: 332.4957213997841
epoch: 464 loss: 332.79939652234316
epoch: 465 loss: 351.2036486417055
epoch: 466 loss: 335.07283433154225
epoch: 467 loss: 329.95471791923046
epoch: 468 loss: 332.99707843363285
epoch: 469 loss: 331.4549261443317
epoch: 470 loss: 336.92952870577574
Train accuracy of the network on the 100000 diagrams: 91.53600 %
Validation accuracy of the network on the 100000 diagrams: 91.35600 %
epoch: 471 loss: 336.85681814327836
epoch: 472 loss: 331.25639916583896
epoch: 473 loss: 331.82981499657035
epoch: 474 loss: 331.44882898777723
epoch: 475 loss: 342.4808977097273
epoch: 476 loss: 331.29571752250195
epoch: 477 loss: 333.7505121231079
epoch: 478 loss: 332.00596490874887
epoch: 479 loss: 333.30945117399096
epoch: 480 loss: 330.21960949897766
Train accuracy of the network on the 100000 diagrams: 91.44300 %
Validation accuracy of the network on the 100000 diagrams: 91.25800 %
epoch: 481 loss: 338.22905529662967
epoch: 482 loss: 332.9480478838086
epoch: 483 loss: 330.58377436921
epoch: 484 loss: 334.5615497045219
epoch: 485 loss: 331.47703678533435
epoch: 486 loss: 330.27075454592705
epoch: 487 loss: 334.6009418591857
epoch: 488 loss: 336.9486421905458
epoch: 489 loss: 328.81575828418136
epoch: 490 loss: 331.30585070326924
Train accuracy of the network on the 100000 diagrams: 91.64000 %
Validation accuracy of the network on the 100000 diagrams: 91.44000 %
epoch: 491 loss: 330.15097710490227
epoch: 492 loss: 332.4700839519501
epoch: 493 loss: 431.2738397344947
epoch: 494 loss: 342.2256827764213
epoch: 495 loss: 332.68580208718777
epoch: 496 loss: 331.03333741798997
epoch: 497 loss: 332.0601937621832
epoch: 498 loss: 330.93277736753225
epoch: 499 loss: 333.01736152917147
epoch: 500 loss: 329.8972711414099
Train accuracy of the network on the 100000 diagrams: 91.12200 %
Validation accuracy of the network on the 100000 diagrams: 91.09100 %
epoch: 501 loss: 331.84910625964403
epoch: 502 loss: 339.0904140025377
epoch: 503 loss: 333.56046161800623
epoch: 504 loss: 328.9326389953494
epoch: 505 loss: 332.506451651454
epoch: 506 loss: 330.8099043406546
epoch: 507 loss: 330.30771321058273
epoch: 508 loss: 357.7998919002712
epoch: 509 loss: 338.74510456994176
epoch: 510 loss: 329.3151816651225
Train accuracy of the network on the 100000 diagrams: 91.40300 %
Validation accuracy of the network on the 100000 diagrams: 91.25000 %
epoch: 511 loss: 329.88814697414637
epoch: 512 loss: 330.927290674299
epoch: 513 loss: 331.7388766296208
epoch: 514 loss: 329.5223277807236
epoch: 515 loss: 330.6255954839289
epoch: 516 loss: 331.6414764560759
epoch: 517 loss: 523.1160583943129
epoch: 518 loss: 441.75769731402397
epoch: 519 loss: 359.6198688521981
epoch: 520 loss: 350.1244448572397
Train accuracy of the network on the 100000 diagrams: 91.21000 %
Validation accuracy of the network on the 100000 diagrams: 91.13800 %
epoch: 521 loss: 552.1678975634277
epoch: 522 loss: 360.73686197400093
epoch: 523 loss: 344.6367614939809
epoch: 524 loss: 345.61166094243526
epoch: 525 loss: 339.31690143048763
epoch: 526 loss: 342.9112230911851
epoch: 527 loss: 337.2933175973594
epoch: 528 loss: 339.4646872021258
epoch: 529 loss: 350.0626570209861
epoch: 530 loss: 333.90831303596497
Train accuracy of the network on the 100000 diagrams: 91.42900 %
Validation accuracy of the network on the 100000 diagrams: 91.35000 %
epoch: 531 loss: 351.79900884255767
epoch: 532 loss: 339.8212771192193
epoch: 533 loss: 333.3544835485518
epoch: 534 loss: 332.0239195525646
epoch: 535 loss: 332.8409828469157
epoch: 536 loss: 330.2920578569174
epoch: 537 loss: 336.35464600101113
epoch: 538 loss: 331.0305828526616
epoch: 539 loss: 329.257151812315
epoch: 540 loss: 337.4959104396403
Train accuracy of the network on the 100000 diagrams: 91.43600 %
Validation accuracy of the network on the 100000 diagrams: 91.28600 %
epoch: 541 loss: 331.1995666399598
epoch: 542 loss: 330.50346929207444
epoch: 543 loss: 331.88462970405817
epoch: 544 loss: 329.74620870128274
epoch: 545 loss: 330.79292511194944
epoch: 546 loss: 329.7139760926366
epoch: 547 loss: 329.6810017861426
epoch: 548 loss: 339.812994107604
epoch: 549 loss: 339.9226720072329
epoch: 550 loss: 329.12274473533034
Train accuracy of the network on the 100000 diagrams: 91.64700 %
Validation accuracy of the network on the 100000 diagrams: 91.43000 %
epoch: 551 loss: 329.5804892703891
epoch: 552 loss: 336.1657730862498
epoch: 553 loss: 332.3265139684081
epoch: 554 loss: 328.4190845750272
epoch: 555 loss: 332.5888088978827
epoch: 556 loss: 329.57195346057415"""
# %%
losses = []
for line in input_text.split("\n"):
    if line.split()[0] == "epoch:":
        loss = float(line.split()[-1])
        if loss < 400:
            losses.append(loss)


coef = np.polyfit(range(len(losses)), losses, 1)
poly1d_fn = np.poly1d(coef) 
#plt.plot(range(len(losses)), losses, 'yo', range(len(losses)), poly1d_fn(losses), '--k')
plt.plot(range(len(losses)), losses)
plt.show()
# %%
