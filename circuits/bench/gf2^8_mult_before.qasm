OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[16];
ccz q[7], q[9], q[16];
h q[16];
h q[16];
ccz q[6], q[10], q[16];
h q[16];
h q[16];
ccz q[5], q[11], q[16];
h q[16];
h q[16];
ccz q[4], q[12], q[16];
h q[16];
h q[16];
ccz q[3], q[13], q[16];
h q[16];
h q[16];
ccz q[2], q[14], q[16];
h q[16];
h q[16];
ccz q[1], q[15], q[16];
h q[16];
h q[17];
ccz q[7], q[10], q[17];
h q[17];
h q[17];
ccz q[6], q[11], q[17];
h q[17];
h q[17];
ccz q[5], q[12], q[17];
h q[17];
h q[17];
ccz q[4], q[13], q[17];
h q[17];
h q[17];
ccz q[3], q[14], q[17];
h q[17];
h q[17];
ccz q[2], q[15], q[17];
h q[17];
h q[18];
ccz q[7], q[11], q[18];
h q[18];
h q[18];
ccz q[6], q[12], q[18];
h q[18];
h q[18];
ccz q[5], q[13], q[18];
h q[18];
h q[18];
ccz q[4], q[14], q[18];
h q[18];
h q[18];
ccz q[3], q[15], q[18];
h q[18];
h q[19];
ccz q[7], q[12], q[19];
h q[19];
h q[19];
ccz q[6], q[13], q[19];
h q[19];
h q[19];
ccz q[5], q[14], q[19];
h q[19];
h q[19];
ccz q[4], q[15], q[19];
h q[19];
h q[20];
ccz q[7], q[13], q[20];
h q[20];
h q[20];
ccz q[6], q[14], q[20];
h q[20];
h q[20];
ccz q[5], q[15], q[20];
h q[20];
h q[21];
ccz q[7], q[14], q[21];
h q[21];
h q[21];
ccz q[6], q[15], q[21];
h q[21];
h q[22];
ccz q[7], q[15], q[22];
h q[22];
cx q[22], q[18];
cx q[22], q[17];
cx q[22], q[16];
cx q[21], q[17];
cx q[21], q[16];
cx q[21], q[23];
cx q[20], q[16];
cx q[20], q[23];
cx q[20], q[22];
cx q[19], q[23];
cx q[19], q[22];
cx q[19], q[21];
cx q[18], q[22];
cx q[18], q[21];
cx q[18], q[20];
cx q[17], q[21];
cx q[17], q[20];
cx q[17], q[19];
cx q[16], q[20];
cx q[16], q[19];
cx q[16], q[18];
h q[23];
ccz q[7], q[8], q[23];
h q[23];
h q[23];
ccz q[6], q[9], q[23];
h q[23];
h q[23];
ccz q[5], q[10], q[23];
h q[23];
h q[23];
ccz q[4], q[11], q[23];
h q[23];
h q[23];
ccz q[3], q[12], q[23];
h q[23];
h q[23];
ccz q[2], q[13], q[23];
h q[23];
h q[23];
ccz q[1], q[14], q[23];
h q[23];
h q[23];
ccz q[0], q[15], q[23];
h q[23];
h q[22];
ccz q[6], q[8], q[22];
h q[22];
h q[22];
ccz q[5], q[9], q[22];
h q[22];
h q[22];
ccz q[4], q[10], q[22];
h q[22];
h q[22];
ccz q[3], q[11], q[22];
h q[22];
h q[22];
ccz q[2], q[12], q[22];
h q[22];
h q[22];
ccz q[1], q[13], q[22];
h q[22];
h q[22];
ccz q[0], q[14], q[22];
h q[22];
h q[21];
ccz q[5], q[8], q[21];
h q[21];
h q[21];
ccz q[4], q[9], q[21];
h q[21];
h q[21];
ccz q[3], q[10], q[21];
h q[21];
h q[21];
ccz q[2], q[11], q[21];
h q[21];
h q[21];
ccz q[1], q[12], q[21];
h q[21];
h q[21];
ccz q[0], q[13], q[21];
h q[21];
h q[20];
ccz q[4], q[8], q[20];
h q[20];
h q[20];
ccz q[3], q[9], q[20];
h q[20];
h q[20];
ccz q[2], q[10], q[20];
h q[20];
h q[20];
ccz q[1], q[11], q[20];
h q[20];
h q[20];
ccz q[0], q[12], q[20];
h q[20];
h q[19];
ccz q[3], q[8], q[19];
h q[19];
h q[19];
ccz q[2], q[9], q[19];
h q[19];
h q[19];
ccz q[1], q[10], q[19];
h q[19];
h q[19];
ccz q[0], q[11], q[19];
h q[19];
h q[18];
ccz q[2], q[8], q[18];
h q[18];
h q[18];
ccz q[1], q[9], q[18];
h q[18];
h q[18];
ccz q[0], q[10], q[18];
h q[18];
h q[17];
ccz q[1], q[8], q[17];
h q[17];
h q[17];
ccz q[0], q[9], q[17];
h q[17];
h q[16];
ccz q[0], q[8], q[16];
h q[16];
