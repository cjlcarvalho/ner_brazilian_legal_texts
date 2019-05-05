#! /bin/bash
wget -rkpN --cut-dirs=3 -e robots=off https://cic.unb.br/~teodecampos/LeNER-Br/leNER-Br/
mv cic.unb.br/* .
rm -rf cic.unb.br index.html* *.gif metadata/ raw_text/ scripts/ dev/index.html* train/index.html* test/index.html*
