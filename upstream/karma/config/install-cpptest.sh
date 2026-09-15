echo "! \tDownloading CPPTest Version=1.1.1\t !"
wget http://sourceforge.net/projects/cpptest/files/cpptest/cpptest-1.1.1/cpptest-1.1.1.tar.gz/download -o download.log
gunzip -df cpptest-1.1.1.tar.gz
tar -xf cpptest-1.1.1.tar
rm cpptest-1.1.1.tar
mv cpptest-1.1.1 cpptest
cd cpptest
echo "! \tConfiguring CPPTest\t !"
./configure --prefix=$PWD/..  > config.log
echo "! \tCompiling and Installing CPPTest\t !"
make all install > make.log
cd ..
