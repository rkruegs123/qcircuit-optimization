OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[1], q[6];
t q[2];
cx q[12], q[1];
cx q[9], q[18];
cx q[16], q[6];
h q[19];
cx q[14], q[1];
cx q[0], q[7];
t q[19];
cx q[16], q[0];
t q[15];
cx q[11], q[18];
cx q[6], q[7];
cx q[6], q[15];
h q[11];
t q[18];
cx q[12], q[3];
cx q[0], q[3];
cx q[10], q[14];
cx q[5], q[4];
t q[13];
cx q[11], q[15];
cx q[1], q[14];
cx q[16], q[8];
cx q[13], q[18];
cx q[12], q[16];
h q[18];
cx q[18], q[3];
t q[19];
t q[7];
cx q[2], q[5];
h q[7];
cx q[15], q[18];
h q[2];
t q[16];
h q[10];
h q[13];
cx q[7], q[0];
cx q[16], q[14];
cx q[7], q[8];
cx q[18], q[8];
cx q[1], q[0];
t q[2];
t q[12];
cx q[12], q[5];
cx q[1], q[2];
h q[18];
t q[14];
h q[1];
t q[9];
t q[18];
cx q[13], q[9];
h q[16];
h q[3];
cx q[10], q[12];
cx q[13], q[8];
cx q[6], q[1];
cx q[15], q[19];
cx q[13], q[15];
cx q[17], q[12];
cx q[16], q[13];
cx q[1], q[9];
cx q[10], q[1];
cx q[2], q[13];
h q[8];
cx q[14], q[1];
cx q[12], q[18];
cx q[5], q[11];
h q[1];
h q[2];
h q[6];
cx q[1], q[6];
cx q[16], q[3];
cx q[9], q[18];
cx q[13], q[17];
t q[15];
cx q[9], q[0];
cx q[12], q[15];
t q[0];
h q[1];
h q[4];
cx q[11], q[3];
cx q[11], q[0];
cx q[18], q[7];
h q[2];
t q[0];
cx q[7], q[8];
t q[13];
cx q[6], q[14];
t q[8];
cx q[11], q[8];
t q[15];
cx q[17], q[7];
t q[18];
cx q[2], q[12];
h q[7];
cx q[6], q[19];
cx q[4], q[6];
h q[3];
cx q[19], q[6];
