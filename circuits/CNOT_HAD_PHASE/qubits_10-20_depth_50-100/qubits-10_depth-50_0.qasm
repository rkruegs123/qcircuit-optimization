OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[4];
t q[0];
cx q[0], q[4];
cx q[1], q[7];
h q[2];
cx q[2], q[7];
h q[6];
h q[9];
cx q[4], q[0];
cx q[7], q[1];
cx q[7], q[5];
cx q[5], q[8];
cx q[7], q[8];
t q[7];
cx q[0], q[5];
cx q[1], q[5];
h q[4];
h q[6];
cx q[4], q[9];
t q[4];
h q[3];
cx q[8], q[2];
cx q[2], q[1];
t q[2];
cx q[1], q[8];
cx q[1], q[4];
h q[7];
cx q[9], q[2];
cx q[4], q[6];
h q[4];
cx q[3], q[2];
t q[0];
cx q[7], q[6];
cx q[0], q[5];
cx q[8], q[1];
cx q[4], q[6];
cx q[3], q[4];
cx q[6], q[1];
cx q[1], q[2];
cx q[4], q[7];
t q[5];
cx q[6], q[0];
cx q[1], q[8];
cx q[3], q[9];
cx q[0], q[4];
cx q[2], q[9];
h q[3];
cx q[6], q[1];
h q[3];
cx q[8], q[3];
