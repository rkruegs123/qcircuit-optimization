OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[7], q[1];
t q[11];
cx q[18], q[7];
cx q[13], q[6];
t q[4];
h q[7];
cx q[19], q[18];
cx q[5], q[6];
cx q[5], q[10];
cx q[13], q[11];
cx q[7], q[17];
cx q[1], q[17];
t q[13];
cx q[12], q[3];
t q[16];
cx q[15], q[2];
t q[6];
cx q[18], q[0];
cx q[10], q[17];
t q[2];
cx q[4], q[8];
t q[5];
h q[12];
cx q[12], q[4];
cx q[0], q[3];
cx q[0], q[14];
cx q[5], q[11];
cx q[4], q[15];
cx q[16], q[3];
t q[16];
t q[3];
cx q[19], q[10];
cx q[14], q[1];
cx q[18], q[5];
cx q[15], q[6];
t q[4];
cx q[2], q[1];
h q[5];
cx q[1], q[18];
h q[19];
t q[18];
cx q[0], q[12];
cx q[3], q[15];
h q[17];
h q[1];
h q[17];
cx q[11], q[12];
cx q[6], q[14];
h q[2];
cx q[13], q[8];