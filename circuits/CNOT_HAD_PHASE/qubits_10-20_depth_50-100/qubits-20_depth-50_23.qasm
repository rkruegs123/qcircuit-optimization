OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[2], q[4];
cx q[17], q[8];
h q[12];
cx q[2], q[16];
t q[18];
cx q[17], q[3];
t q[6];
h q[2];
h q[13];
cx q[9], q[14];
cx q[7], q[11];
t q[1];
cx q[2], q[8];
cx q[10], q[17];
h q[4];
cx q[4], q[7];
h q[2];
cx q[16], q[8];
h q[11];
cx q[17], q[12];
h q[7];
cx q[14], q[6];
cx q[13], q[0];
t q[0];
cx q[1], q[13];
h q[16];
cx q[18], q[13];
cx q[1], q[9];
h q[0];
t q[19];
t q[7];
cx q[11], q[9];
h q[18];
h q[17];
cx q[5], q[18];
cx q[4], q[9];
h q[1];
cx q[3], q[13];
t q[11];
cx q[9], q[18];
cx q[3], q[18];
cx q[11], q[0];
cx q[4], q[8];
t q[16];
cx q[5], q[0];
cx q[16], q[15];
cx q[8], q[1];
cx q[1], q[14];
cx q[3], q[17];
cx q[13], q[10];
