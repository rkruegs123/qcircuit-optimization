OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[6], q[17];
cx q[7], q[18];
h q[13];
cx q[8], q[17];
cx q[11], q[18];
cx q[15], q[11];
cx q[16], q[2];
t q[15];
cx q[2], q[15];
cx q[8], q[9];
h q[14];
h q[15];
h q[16];
t q[6];
cx q[9], q[15];
cx q[8], q[5];
t q[5];
t q[7];
t q[11];
cx q[17], q[3];
h q[15];
cx q[14], q[17];
cx q[3], q[13];
t q[19];
cx q[6], q[9];
cx q[19], q[4];
h q[19];
cx q[8], q[5];
h q[11];
h q[11];
h q[17];
h q[8];
h q[13];
cx q[12], q[16];
cx q[10], q[5];
cx q[12], q[16];
h q[18];
h q[17];
cx q[10], q[13];
t q[0];
h q[8];
h q[1];
cx q[9], q[13];
h q[12];
h q[17];
cx q[18], q[12];
t q[17];
t q[17];
cx q[3], q[11];
t q[12];
cx q[1], q[16];
cx q[17], q[1];
cx q[8], q[16];
t q[15];
cx q[10], q[4];
h q[0];
cx q[10], q[9];
t q[17];
cx q[10], q[12];
cx q[11], q[2];
cx q[17], q[7];
cx q[2], q[0];
cx q[11], q[16];
cx q[15], q[7];
cx q[13], q[5];
cx q[7], q[17];
h q[19];
h q[0];
cx q[2], q[11];
cx q[2], q[4];
cx q[8], q[7];
cx q[11], q[4];
t q[10];
t q[18];
cx q[8], q[0];
cx q[11], q[0];
cx q[10], q[5];
cx q[17], q[0];
cx q[19], q[14];
cx q[13], q[9];
t q[5];
h q[13];
t q[6];
h q[12];
cx q[16], q[2];
t q[7];
cx q[7], q[17];
h q[12];
h q[3];
h q[7];
t q[19];
cx q[17], q[18];
h q[7];
cx q[7], q[16];
cx q[16], q[10];
h q[13];
cx q[1], q[15];
cx q[8], q[11];
t q[2];
t q[9];
