# -*- coding: utf-8 -*-
# 按依赖顺序运行全部程序
import os
import time
import runpy

# 顺序即依赖：innov_model 必须先于 plots
SCRIPTS = [
    'data_analysis.py', 'q1.py', 'q2.py', 'q3.py', 'q4.py',
    'checks.py', 'verify_2d.py',
    'innov_model.py', 'plots.py', 'innov_study.py',
]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    for s in SCRIPTS:
        print('=' * 70)
        print('运行 %s ……' % s, flush=True)
        t1 = time.time()
        try:
            runpy.run_path(s, run_name='__main__')
        except SystemExit as e:
            if e.code not in (None, 0):
                print('%s 退出码 %s，继续后续' % (s, e.code), flush=True)
        except Exception as e:
            print('%s 失败：%r' % (s, e), flush=True)
        print('%s 完成，用时 %.1f s' % (s, time.time() - t1), flush=True)
    print('全部完成，总用时 %.1f min' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    main()