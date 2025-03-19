/* RISC-V opcode list
   Copyright (C) 2011-2020 Free Software Foundation, Inc.

   Contributed by Andrew Waterman (andrew@sifive.com).
   Based on MIPS target.

   PULP family support contributed by Eric Flamand (eflamand@iis.ee.ethz.ch) at ETH-Zurich
   and Greenwaves Technologies (eric.flamand@greenwaves-technologies.com)

   This file is part of the GNU opcodes library.

   This library is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   It is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; see the file COPYING3. If not,
   see <http://www.gnu.org/licenses/>.  */

#include "sysdep.h"
#include "opcode/riscv.h"
#include <stdio.h>

/* Register names used by gas and objdump.  */

const char * const riscv_gpr_names_numeric[NGPR] =
{
  "x0",   "x1",   "x2",   "x3",   "x4",   "x5",   "x6",   "x7",
  "x8",   "x9",   "x10",  "x11",  "x12",  "x13",  "x14",  "x15",
  "x16",  "x17",  "x18",  "x19",  "x20",  "x21",  "x22",  "x23",
  "x24",  "x25",  "x26",  "x27",  "x28",  "x29",  "x30",  "x31"
};

const char * const riscv_gpr_names_abi[NGPR] = {
  "zero", "ra", "sp",  "gp",  "tp", "t0",  "t1",  "t2",
  "s0",   "s1", "a0",  "a1",  "a2", "a3",  "a4",  "a5",
  "a6",   "a7", "s2",  "s3",  "s4", "s5",  "s6",  "s7",
  "s8",   "s9", "s10", "s11", "t3", "t4",  "t5",  "t6"
};

const char * const riscv_fpr_names_numeric[NFPR] =
{
  "f0",   "f1",   "f2",   "f3",   "f4",   "f5",   "f6",   "f7",
  "f8",   "f9",   "f10",  "f11",  "f12",  "f13",  "f14",  "f15",
  "f16",  "f17",  "f18",  "f19",  "f20",  "f21",  "f22",  "f23",
  "f24",  "f25",  "f26",  "f27",  "f28",  "f29",  "f30",  "f31"
};

const char * const riscv_fpr_names_abi[NFPR] = {
  "ft0", "ft1", "ft2",  "ft3",  "ft4", "ft5", "ft6",  "ft7",
  "fs0", "fs1", "fa0",  "fa1",  "fa2", "fa3", "fa4",  "fa5",
  "fa6", "fa7", "fs2",  "fs3",  "fs4", "fs5", "fs6",  "fs7",
  "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"
};

/* The order of overloaded instructions matters.  Label arguments and
   register arguments look the same. Instructions that can have either
   for arguments must apear in the correct order in this table for the
   assembler to pick the right one. In other words, entries with
   immediate operands must apear after the same instruction with
   registers.

   Because of the lookup algorithm used, entries with the same opcode
   name must be contiguous.  */

#define MASK_RS1 (OP_MASK_RS1 << OP_SH_RS1)
#define MASK_RS2 (OP_MASK_RS2 << OP_SH_RS2)
#define MASK_RD (OP_MASK_RD << OP_SH_RD)
#define MASK_CRS2 (OP_MASK_CRS2 << OP_SH_CRS2)
#define MASK_IMM ENCODE_ITYPE_IMM (-1U)
#define MASK_RVC_IMM ENCODE_RVC_IMM (-1U)
#define MASK_UIMM ENCODE_UTYPE_IMM (-1U)
#define MASK_RM (OP_MASK_RM << OP_SH_RM)
#define MASK_PRED (OP_MASK_PRED << OP_SH_PRED)
#define MASK_SUCC (OP_MASK_SUCC << OP_SH_SUCC)
#define MASK_AQ (OP_MASK_AQ << OP_SH_AQ)
#define MASK_RL (OP_MASK_RL << OP_SH_RL)
#define MASK_AQRL (MASK_AQ | MASK_RL)

static int
match_opcode (const struct riscv_opcode *op, insn_t insn)
{
  return ((insn ^ op->match) & op->mask) == 0;
}

static int
match_never (const struct riscv_opcode *op ATTRIBUTE_UNUSED,
	     insn_t insn ATTRIBUTE_UNUSED)
{
  return 0;
}

static int
match_rs1_eq_rs2 (const struct riscv_opcode *op, insn_t insn)
{
  int rs1 = (insn & MASK_RS1) >> OP_SH_RS1;
  int rs2 = (insn & MASK_RS2) >> OP_SH_RS2;
  return match_opcode (op, insn) && rs1 == rs2;
}

static int
match_rd_nonzero (const struct riscv_opcode *op, insn_t insn)
{
  return match_opcode (op, insn) && ((insn & MASK_RD) != 0);
}

static int
match_c_add (const struct riscv_opcode *op, insn_t insn)
{
  return match_rd_nonzero (op, insn) && ((insn & MASK_CRS2) != 0);
}

/* We don't allow mv zero,X to become a c.mv hint, so we need a separate
   matching function for this.  */

static int
match_c_add_with_hint (const struct riscv_opcode *op, insn_t insn)
{
  return match_opcode (op, insn) && ((insn & MASK_CRS2) != 0);
}

static int
match_c_nop (const struct riscv_opcode *op, insn_t insn)
{
  return (match_opcode (op, insn)
	  && (((insn & MASK_RD) >> OP_SH_RD) == 0));
}

static int
match_c_addi16sp (const struct riscv_opcode *op, insn_t insn)
{
  return (match_opcode (op, insn)
	  && (((insn & MASK_RD) >> OP_SH_RD) == 2)
	  && EXTRACT_RVC_ADDI16SP_IMM (insn) != 0);
}

static int
match_c_lui (const struct riscv_opcode *op, insn_t insn)
{
  return (match_rd_nonzero (op, insn)
	  && (((insn & MASK_RD) >> OP_SH_RD) != 2)
	  && EXTRACT_RVC_LUI_IMM (insn) != 0);
}

/* We don't allow lui zero,X to become a c.lui hint, so we need a separate
   matching function for this.  */

static int
match_c_lui_with_hint (const struct riscv_opcode *op, insn_t insn)
{
  return (match_opcode (op, insn)
	  && (((insn & MASK_RD) >> OP_SH_RD) != 2)
	  && EXTRACT_RVC_LUI_IMM (insn) != 0);
}

static int
match_c_addi4spn (const struct riscv_opcode *op, insn_t insn)
{
  return match_opcode (op, insn) && EXTRACT_RVC_ADDI4SPN_IMM (insn) != 0;
}

/* This requires a non-zero shift.  A zero rd is a hint, so is allowed.  */

static int
match_c_slli (const struct riscv_opcode *op, insn_t insn)
{
  return match_opcode (op, insn) && EXTRACT_RVC_IMM (insn) != 0;
}

/* This requires a non-zero rd, and a non-zero shift.  */

static int
match_slli_as_c_slli (const struct riscv_opcode *op, insn_t insn)
{
  return match_rd_nonzero (op, insn) && EXTRACT_RVC_IMM (insn) != 0;
}

/* This requires a zero shift.  A zero rd is a hint, so is allowed.  */

static int
match_c_slli64 (const struct riscv_opcode *op, insn_t insn)
{
  return match_opcode (op, insn) && EXTRACT_RVC_IMM (insn) == 0;
}

/* This is used for both srli and srai.  This requires a non-zero shift.
   A zero rd is not possible.  */

static int
match_srxi_as_c_srxi (const struct riscv_opcode *op, insn_t insn)
{
  return match_opcode (op, insn) && EXTRACT_RVC_IMM (insn) != 0;
}

const struct riscv_opcode riscv_opcodes[] =
{
/* name,     xlen, isa,   operands, match, mask, match_func, pinfo.  */
{"cmul",  	0, INSN_CLASS_I,   "s,t",  MATCH_CMUL, MASK_CMUL, match_opcode, 0 },
{"cget",  	0, INSN_CLASS_I,   "d",  MATCH_CGET, MASK_CGET, match_opcode, 0 },
{"crst",  	0, INSN_CLASS_I,   "",  MATCH_CRST, MASK_CRST, match_opcode, 0 },
{"unimp",       0, INSN_CLASS_C,   "",  0, 0xffffU,  match_opcode, INSN_ALIAS },
{"unimp",       0, INSN_CLASS_I,   "",  MATCH_CSRRW | (CSR_CYCLE << OP_SH_CSR), 0xffffffffU,  match_opcode, 0 }, /* csrw cycle, x0 */
{"ebreak",      0, INSN_CLASS_C,   "",  MATCH_C_EBREAK, MASK_C_EBREAK, match_opcode, INSN_ALIAS },
{"ebreak",      0, INSN_CLASS_I,   "",    MATCH_EBREAK, MASK_EBREAK, match_opcode, 0 },
{"sbreak",      0, INSN_CLASS_C,   "",  MATCH_C_EBREAK, MASK_C_EBREAK, match_opcode, INSN_ALIAS },
{"sbreak",      0, INSN_CLASS_I,   "",    MATCH_EBREAK, MASK_EBREAK, match_opcode, INSN_ALIAS },
{"ret",         0, INSN_CLASS_C,   "",  MATCH_C_JR | (X_RA << OP_SH_RD), MASK_C_JR | MASK_RD, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"ret",         0, INSN_CLASS_I,   "",  MATCH_JALR | (X_RA << OP_SH_RS1), MASK_JALR | MASK_RD | MASK_RS1 | MASK_IMM, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"jr",          0, INSN_CLASS_C,   "d",  MATCH_C_JR, MASK_C_JR, match_rd_nonzero, INSN_ALIAS|INSN_BRANCH },
{"jr",          0, INSN_CLASS_I,   "s",  MATCH_JALR, MASK_JALR | MASK_RD | MASK_IMM, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"jr",          0, INSN_CLASS_I,   "o(s)",  MATCH_JALR, MASK_JALR | MASK_RD, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"jr",          0, INSN_CLASS_I,   "s,j",  MATCH_JALR, MASK_JALR | MASK_RD, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"jalr",        0, INSN_CLASS_C,   "d",  MATCH_C_JALR, MASK_C_JALR, match_rd_nonzero, INSN_ALIAS|INSN_JSR },
{"jalr",        0, INSN_CLASS_I,   "s",  MATCH_JALR | (X_RA << OP_SH_RD), MASK_JALR | MASK_RD | MASK_IMM, match_opcode, INSN_ALIAS|INSN_JSR },
{"jalr",        0, INSN_CLASS_I,   "o(s)",  MATCH_JALR | (X_RA << OP_SH_RD), MASK_JALR | MASK_RD, match_opcode, INSN_ALIAS|INSN_JSR },
{"jalr",        0, INSN_CLASS_I,   "s,j",  MATCH_JALR | (X_RA << OP_SH_RD), MASK_JALR | MASK_RD, match_opcode, INSN_ALIAS|INSN_JSR },
{"jalr",        0, INSN_CLASS_I,   "d,s",  MATCH_JALR, MASK_JALR | MASK_IMM, match_opcode, INSN_ALIAS|INSN_JSR },
{"jalr",        0, INSN_CLASS_I,   "d,o(s)",  MATCH_JALR, MASK_JALR, match_opcode, INSN_JSR },
{"jalr",        0, INSN_CLASS_I,   "d,s,j",  MATCH_JALR, MASK_JALR, match_opcode, INSN_JSR },
{"j",           0, INSN_CLASS_C,   "Ca",  MATCH_C_J, MASK_C_J, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"j",           0, INSN_CLASS_I,   "a",  MATCH_JAL, MASK_JAL | MASK_RD, match_opcode, INSN_ALIAS|INSN_BRANCH },
{"jal",         0, INSN_CLASS_I,   "d,a",  MATCH_JAL, MASK_JAL, match_opcode, INSN_JSR },
{"jal",        32, INSN_CLASS_C,   "Ca",  MATCH_C_JAL, MASK_C_JAL, match_opcode, INSN_ALIAS|INSN_JSR },
{"jal",         0, INSN_CLASS_I,   "a",  MATCH_JAL | (X_RA << OP_SH_RD), MASK_JAL | MASK_RD, match_opcode, INSN_ALIAS|INSN_JSR },
{"call",        0, INSN_CLASS_I,   "d,c", (X_T1 << OP_SH_RS1), (int) M_CALL,  match_never, INSN_MACRO },
{"call",        0, INSN_CLASS_I,   "c", (X_RA << OP_SH_RS1) | (X_RA << OP_SH_RD), (int) M_CALL,  match_never, INSN_MACRO },
{"tail",        0, INSN_CLASS_I,   "c", (X_T1 << OP_SH_RS1), (int) M_CALL,  match_never, INSN_MACRO },
{"jump",        0, INSN_CLASS_I,   "c,s", 0, (int) M_CALL,  match_never, INSN_MACRO },
{"nop",         0, INSN_CLASS_C,   "",  MATCH_C_ADDI, 0xffff, match_opcode, INSN_ALIAS },
{"nop",         0, INSN_CLASS_I,   "",         MATCH_ADDI, MASK_ADDI | MASK_RD | MASK_RS1 | MASK_IMM, match_opcode, INSN_ALIAS },
{"lui",         0, INSN_CLASS_C,   "d,Cu",  MATCH_C_LUI, MASK_C_LUI, match_c_lui, INSN_ALIAS },
{"lui",         0, INSN_CLASS_I,   "d,u",  MATCH_LUI, MASK_LUI, match_opcode, 0 },
{"li",          0, INSN_CLASS_C,   "d,Cv",  MATCH_C_LUI, MASK_C_LUI, match_c_lui, INSN_ALIAS },
{"li",          0, INSN_CLASS_C,   "d,Co",  MATCH_C_LI, MASK_C_LI, match_rd_nonzero, INSN_ALIAS },
{"li",          0, INSN_CLASS_I,   "d,j",      MATCH_ADDI, MASK_ADDI | MASK_RS1, match_opcode, INSN_ALIAS }, /* addi */
{"li",          0, INSN_CLASS_I,   "d,I",  0,    (int) M_LI,  match_never, INSN_MACRO },
{"mv",          0, INSN_CLASS_C,   "d,CV",  MATCH_C_MV, MASK_C_MV, match_c_add, INSN_ALIAS },
{"mv",          0, INSN_CLASS_I,   "d,s",  MATCH_ADDI, MASK_ADDI | MASK_IMM, match_opcode, INSN_ALIAS },
{"move",        0, INSN_CLASS_C,   "d,CV",  MATCH_C_MV, MASK_C_MV, match_c_add, INSN_ALIAS },
{"move",        0, INSN_CLASS_I,   "d,s",  MATCH_ADDI, MASK_ADDI | MASK_IMM, match_opcode, INSN_ALIAS },
{"andi",        0, INSN_CLASS_C,   "Cs,Cw,Co",  MATCH_C_ANDI, MASK_C_ANDI, match_opcode, INSN_ALIAS },
{"andi",        0, INSN_CLASS_I,   "d,s,j",  MATCH_ANDI, MASK_ANDI, match_opcode, 0 },
{"and",         0, INSN_CLASS_C,   "Cs,Cw,Ct",  MATCH_C_AND, MASK_C_AND, match_opcode, INSN_ALIAS },
{"and",         0, INSN_CLASS_C,   "Cs,Ct,Cw",  MATCH_C_AND, MASK_C_AND, match_opcode, INSN_ALIAS },
{"and",         0, INSN_CLASS_C,   "Cs,Cw,Co",  MATCH_C_ANDI, MASK_C_ANDI, match_opcode, INSN_ALIAS },
{"and",         0, INSN_CLASS_I,   "d,s,t",  MATCH_AND, MASK_AND, match_opcode, 0 },
{"and",         0, INSN_CLASS_I,   "d,s,j",  MATCH_ANDI, MASK_ANDI, match_opcode, INSN_ALIAS },
{"beqz",        0, INSN_CLASS_C,   "Cs,Cp",  MATCH_C_BEQZ, MASK_C_BEQZ, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"beqz",        0, INSN_CLASS_I,   "s,p",  MATCH_BEQ, MASK_BEQ | MASK_RS2, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"beq",         0, INSN_CLASS_C,   "Cs,Cz,Cp",  MATCH_C_BEQZ, MASK_C_BEQZ, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"beq",         0, INSN_CLASS_I,   "s,t,p",  MATCH_BEQ, MASK_BEQ, match_opcode, INSN_CONDBRANCH },
{"blez",        0, INSN_CLASS_I,   "t,p",  MATCH_BGE, MASK_BGE | MASK_RS1, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bgez",        0, INSN_CLASS_I,   "s,p",  MATCH_BGE, MASK_BGE | MASK_RS2, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bge",         0, INSN_CLASS_I,   "s,t,p",  MATCH_BGE, MASK_BGE, match_opcode, INSN_CONDBRANCH },
{"bgeu",        0, INSN_CLASS_I,   "s,t,p",  MATCH_BGEU, MASK_BGEU, match_opcode, INSN_CONDBRANCH },
{"ble",         0, INSN_CLASS_I,   "t,s,p",  MATCH_BGE, MASK_BGE, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bleu",        0, INSN_CLASS_I,   "t,s,p",  MATCH_BGEU, MASK_BGEU, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bltz",        0, INSN_CLASS_I,   "s,p",  MATCH_BLT, MASK_BLT | MASK_RS2, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bgtz",        0, INSN_CLASS_I,   "t,p",  MATCH_BLT, MASK_BLT | MASK_RS1, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"blt",         0, INSN_CLASS_I,   "s,t,p",  MATCH_BLT, MASK_BLT, match_opcode, INSN_CONDBRANCH },
{"bltu",        0, INSN_CLASS_I,   "s,t,p",  MATCH_BLTU, MASK_BLTU, match_opcode, INSN_CONDBRANCH },
{"bgt",         0, INSN_CLASS_I,   "t,s,p",  MATCH_BLT, MASK_BLT, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bgtu",        0, INSN_CLASS_I,   "t,s,p",  MATCH_BLTU, MASK_BLTU, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bnez",        0, INSN_CLASS_C,   "Cs,Cp",  MATCH_C_BNEZ, MASK_C_BNEZ, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bnez",        0, INSN_CLASS_I,   "s,p",  MATCH_BNE, MASK_BNE | MASK_RS2, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bne",         0, INSN_CLASS_C,   "Cs,Cz,Cp",  MATCH_C_BNEZ, MASK_C_BNEZ, match_opcode, INSN_ALIAS|INSN_CONDBRANCH },
{"bne",         0, INSN_CLASS_I,   "s,t,p",  MATCH_BNE, MASK_BNE, match_opcode, INSN_CONDBRANCH },
{"addi",        0, INSN_CLASS_C,   "Ct,Cc,CK", MATCH_C_ADDI4SPN, MASK_C_ADDI4SPN, match_c_addi4spn, INSN_ALIAS },
{"addi",        0, INSN_CLASS_C,   "d,CU,Cj",  MATCH_C_ADDI, MASK_C_ADDI, match_rd_nonzero, INSN_ALIAS },
{"addi",        0, INSN_CLASS_C,   "d,CU,z",    MATCH_C_NOP, MASK_C_ADDI | MASK_RVC_IMM, match_c_nop, INSN_ALIAS },
{"addi",        0, INSN_CLASS_C,   "Cc,Cc,CL", MATCH_C_ADDI16SP, MASK_C_ADDI16SP, match_c_addi16sp, INSN_ALIAS },
{"addi",        0, INSN_CLASS_I,   "d,s,j",  MATCH_ADDI, MASK_ADDI, match_opcode, 0 },
{"add",         0, INSN_CLASS_C,   "d,CU,CV",  MATCH_C_ADD, MASK_C_ADD, match_c_add, INSN_ALIAS },
{"add",         0, INSN_CLASS_C,   "d,CV,CU",  MATCH_C_ADD, MASK_C_ADD, match_c_add, INSN_ALIAS },
{"add",         0, INSN_CLASS_C,   "d,CU,Co",  MATCH_C_ADDI, MASK_C_ADDI, match_rd_nonzero, INSN_ALIAS },
{"add",         0, INSN_CLASS_C,   "Ct,Cc,CK", MATCH_C_ADDI4SPN, MASK_C_ADDI4SPN, match_c_addi4spn, INSN_ALIAS },
{"add",         0, INSN_CLASS_C,   "Cc,Cc,CL", MATCH_C_ADDI16SP, MASK_C_ADDI16SP, match_c_addi16sp, INSN_ALIAS },
{"add",         0, INSN_CLASS_I,   "d,s,t",  MATCH_ADD, MASK_ADD, match_opcode, 0 },
/* This is used for TLS, where the fourth arg is %tprel_add, to get a reloc
   applied to an add instruction, for relaxation to use.  */
{"add",         0, INSN_CLASS_I,   "d,s,t,1",MATCH_ADD, MASK_ADD, match_opcode, 0 },
{"add",         0, INSN_CLASS_I,   "d,s,j",  MATCH_ADDI, MASK_ADDI, match_opcode, INSN_ALIAS },
{"la",          0, INSN_CLASS_I,   "d,B",  0,    (int) M_LA,  match_never, INSN_MACRO },
{"lla",         0, INSN_CLASS_I,   "d,B",  0,    (int) M_LLA,  match_never, INSN_MACRO },
{"la.tls.gd",   0, INSN_CLASS_I,   "d,A",  0,    (int) M_LA_TLS_GD,  match_never, INSN_MACRO },
{"la.tls.ie",   0, INSN_CLASS_I,   "d,A",  0,    (int) M_LA_TLS_IE,  match_never, INSN_MACRO },
{"neg",         0, INSN_CLASS_I,   "d,t",  MATCH_SUB, MASK_SUB | MASK_RS1, match_opcode, INSN_ALIAS }, /* sub 0 */
{"slli",        0, INSN_CLASS_C,   "d,CU,C>",  MATCH_C_SLLI, MASK_C_SLLI, match_slli_as_c_slli, INSN_ALIAS },
{"slli",        0, INSN_CLASS_I,   "d,s,>",   MATCH_SLLI, MASK_SLLI, match_opcode, 0 },
{"sll",         0, INSN_CLASS_C,   "d,CU,C>",  MATCH_C_SLLI, MASK_C_SLLI, match_slli_as_c_slli, INSN_ALIAS },
{"sll",         0, INSN_CLASS_I,   "d,s,t",   MATCH_SLL, MASK_SLL, match_opcode, 0 },
{"sll",         0, INSN_CLASS_I,   "d,s,>",   MATCH_SLLI, MASK_SLLI, match_opcode, INSN_ALIAS },
{"srli",        0, INSN_CLASS_C,   "Cs,Cw,C>",  MATCH_C_SRLI, MASK_C_SRLI, match_srxi_as_c_srxi, INSN_ALIAS },
{"srli",        0, INSN_CLASS_I,   "d,s,>",   MATCH_SRLI, MASK_SRLI, match_opcode, 0 },
{"srl",         0, INSN_CLASS_C,   "Cs,Cw,C>",  MATCH_C_SRLI, MASK_C_SRLI, match_srxi_as_c_srxi, INSN_ALIAS },
{"srl",         0, INSN_CLASS_I,   "d,s,t",   MATCH_SRL, MASK_SRL, match_opcode, 0 },
{"srl",         0, INSN_CLASS_I,   "d,s,>",   MATCH_SRLI, MASK_SRLI, match_opcode, INSN_ALIAS },
{"srai",        0, INSN_CLASS_C,   "Cs,Cw,C>",  MATCH_C_SRAI, MASK_C_SRAI, match_srxi_as_c_srxi, INSN_ALIAS },
{"srai",        0, INSN_CLASS_I,   "d,s,>",   MATCH_SRAI, MASK_SRAI, match_opcode, 0 },
{"sra",         0, INSN_CLASS_C,   "Cs,Cw,C>",  MATCH_C_SRAI, MASK_C_SRAI, match_srxi_as_c_srxi, INSN_ALIAS },
{"sra",         0, INSN_CLASS_I,   "d,s,t",   MATCH_SRA, MASK_SRA, match_opcode, 0 },
{"sra",         0, INSN_CLASS_I,   "d,s,>",   MATCH_SRAI, MASK_SRAI, match_opcode, INSN_ALIAS },
{"sub",         0, INSN_CLASS_C,   "Cs,Cw,Ct",  MATCH_C_SUB, MASK_C_SUB, match_opcode, INSN_ALIAS },
{"sub",         0, INSN_CLASS_I,   "d,s,t",  MATCH_SUB, MASK_SUB, match_opcode, 0 },
{"lb",          0, INSN_CLASS_I,   "d,o(s)",  MATCH_LB, MASK_LB, match_opcode, INSN_DREF|INSN_1_BYTE },
{"lb",          0, INSN_CLASS_I,   "d,A",  0, (int) M_LB, match_never, INSN_MACRO },
{"lbu",         0, INSN_CLASS_I,   "d,o(s)",  MATCH_LBU, MASK_LBU, match_opcode, INSN_DREF|INSN_1_BYTE },
{"lbu",         0, INSN_CLASS_I,   "d,A",  0, (int) M_LBU, match_never, INSN_MACRO },
{"lh",          0, INSN_CLASS_I,   "d,o(s)",  MATCH_LH, MASK_LH, match_opcode, INSN_DREF|INSN_2_BYTE },
{"lh",          0, INSN_CLASS_I,   "d,A",  0, (int) M_LH, match_never, INSN_MACRO },
{"lhu",         0, INSN_CLASS_I,   "d,o(s)",  MATCH_LHU, MASK_LHU, match_opcode, INSN_DREF|INSN_2_BYTE },
{"lhu",         0, INSN_CLASS_I,   "d,A",  0, (int) M_LHU, match_never, INSN_MACRO },
{"lw",          0, INSN_CLASS_C,   "d,Cm(Cc)",  MATCH_C_LWSP, MASK_C_LWSP, match_rd_nonzero, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"lw",          0, INSN_CLASS_C,   "Ct,Ck(Cs)",  MATCH_C_LW, MASK_C_LW, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"lw",          0, INSN_CLASS_I,   "d,o(s)",  MATCH_LW, MASK_LW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"lw",          0, INSN_CLASS_I,   "d,A",  0, (int) M_LW, match_never, INSN_MACRO },
{"not",         0, INSN_CLASS_I,   "d,s",  MATCH_XORI | MASK_IMM, MASK_XORI | MASK_IMM, match_opcode, INSN_ALIAS },
{"ori",         0, INSN_CLASS_I,   "d,s,j",  MATCH_ORI, MASK_ORI, match_opcode, 0 },
{"or",          0, INSN_CLASS_C,   "Cs,Cw,Ct",  MATCH_C_OR, MASK_C_OR, match_opcode, INSN_ALIAS },
{"or",          0, INSN_CLASS_C,   "Cs,Ct,Cw",  MATCH_C_OR, MASK_C_OR, match_opcode, INSN_ALIAS },
{"or",          0, INSN_CLASS_I,   "d,s,t",  MATCH_OR, MASK_OR, match_opcode, 0 },
{"or",          0, INSN_CLASS_I,   "d,s,j",  MATCH_ORI, MASK_ORI, match_opcode, INSN_ALIAS },
{"auipc",       0, INSN_CLASS_I,   "d,u",  MATCH_AUIPC, MASK_AUIPC, match_opcode, 0 },
{"seqz",        0, INSN_CLASS_I,   "d,s",  MATCH_SLTIU | ENCODE_ITYPE_IMM (1), MASK_SLTIU | MASK_IMM, match_opcode, INSN_ALIAS },
{"snez",        0, INSN_CLASS_I,   "d,t",  MATCH_SLTU, MASK_SLTU | MASK_RS1, match_opcode, INSN_ALIAS },
{"sltz",        0, INSN_CLASS_I,   "d,s",  MATCH_SLT, MASK_SLT | MASK_RS2, match_opcode, INSN_ALIAS },
{"sgtz",        0, INSN_CLASS_I,   "d,t",  MATCH_SLT, MASK_SLT | MASK_RS1, match_opcode, INSN_ALIAS },
{"slti",        0, INSN_CLASS_I,   "d,s,j",  MATCH_SLTI, MASK_SLTI, match_opcode, 0 },
{"slt",         0, INSN_CLASS_I,   "d,s,t",  MATCH_SLT, MASK_SLT, match_opcode, 0 },
{"slt",         0, INSN_CLASS_I,   "d,s,j",  MATCH_SLTI, MASK_SLTI, match_opcode, INSN_ALIAS },
{"sltiu",       0, INSN_CLASS_I,   "d,s,j",  MATCH_SLTIU, MASK_SLTIU, match_opcode, 0 },
{"sltu",        0, INSN_CLASS_I,   "d,s,t",  MATCH_SLTU, MASK_SLTU, match_opcode, 0 },
{"sltu",        0, INSN_CLASS_I,   "d,s,j",  MATCH_SLTIU, MASK_SLTIU, match_opcode, INSN_ALIAS },
{"sgt",         0, INSN_CLASS_I,   "d,t,s",  MATCH_SLT, MASK_SLT, match_opcode, INSN_ALIAS },
{"sgtu",        0, INSN_CLASS_I,   "d,t,s",  MATCH_SLTU, MASK_SLTU, match_opcode, INSN_ALIAS },
{"sb",          0, INSN_CLASS_I,   "t,q(s)",  MATCH_SB, MASK_SB, match_opcode, INSN_DREF|INSN_1_BYTE },
{"sb",          0, INSN_CLASS_I,   "t,A,s",  0, (int) M_SB, match_never, INSN_MACRO },
{"sh",          0, INSN_CLASS_I,   "t,q(s)",  MATCH_SH, MASK_SH, match_opcode, INSN_DREF|INSN_2_BYTE },
{"sh",          0, INSN_CLASS_I,   "t,A,s",  0, (int) M_SH, match_never, INSN_MACRO },
{"sw",          0, INSN_CLASS_C,   "CV,CM(Cc)",  MATCH_C_SWSP, MASK_C_SWSP, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"sw",          0, INSN_CLASS_C,   "Ct,Ck(Cs)",  MATCH_C_SW, MASK_C_SW, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"sw",          0, INSN_CLASS_I,   "t,q(s)",  MATCH_SW, MASK_SW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"sw",          0, INSN_CLASS_I,   "t,A,s",  0, (int) M_SW, match_never, INSN_MACRO },
{"fence",       0, INSN_CLASS_I,   "",  MATCH_FENCE | MASK_PRED | MASK_SUCC, MASK_FENCE | MASK_RD | MASK_RS1 | MASK_IMM, match_opcode, INSN_ALIAS },
{"fence",       0, INSN_CLASS_I,   "P,Q",  MATCH_FENCE, MASK_FENCE | MASK_RD | MASK_RS1 | (MASK_IMM & ~MASK_PRED & ~MASK_SUCC), match_opcode, 0 },
{"fence.i",     0, INSN_CLASS_I,   "",  MATCH_FENCE_I, MASK_FENCE | MASK_RD | MASK_RS1 | MASK_IMM, match_opcode, 0 },
{"fence.tso",   0, INSN_CLASS_I,   "",  MATCH_FENCE_TSO, MASK_FENCE_TSO | MASK_RD | MASK_RS1, match_opcode, INSN_ALIAS },
{"rdcycle",     0, INSN_CLASS_I,   "d",  MATCH_RDCYCLE, MASK_RDCYCLE, match_opcode, INSN_ALIAS },
{"rdinstret",   0, INSN_CLASS_I,   "d",  MATCH_RDINSTRET, MASK_RDINSTRET, match_opcode, INSN_ALIAS },
{"rdtime",      0, INSN_CLASS_I,   "d",  MATCH_RDTIME, MASK_RDTIME, match_opcode, INSN_ALIAS },
{"rdcycleh",   32, INSN_CLASS_I,   "d",  MATCH_RDCYCLEH, MASK_RDCYCLEH, match_opcode, INSN_ALIAS },
{"rdinstreth", 32, INSN_CLASS_I,   "d",  MATCH_RDINSTRETH, MASK_RDINSTRETH, match_opcode, INSN_ALIAS },
{"rdtimeh",    32, INSN_CLASS_I,   "d",  MATCH_RDTIMEH, MASK_RDTIMEH, match_opcode, INSN_ALIAS },
{"ecall",       0, INSN_CLASS_I,   "",    MATCH_SCALL, MASK_SCALL, match_opcode, 0 },
{"scall",       0, INSN_CLASS_I,   "",    MATCH_SCALL, MASK_SCALL, match_opcode, 0 },
{"xori",        0, INSN_CLASS_I,   "d,s,j",  MATCH_XORI, MASK_XORI, match_opcode, 0 },
{"xor",         0, INSN_CLASS_C,   "Cs,Cw,Ct",  MATCH_C_XOR, MASK_C_XOR, match_opcode, INSN_ALIAS },
{"xor",         0, INSN_CLASS_C,   "Cs,Ct,Cw",  MATCH_C_XOR, MASK_C_XOR, match_opcode, INSN_ALIAS },
{"xor",         0, INSN_CLASS_I,   "d,s,t",  MATCH_XOR, MASK_XOR, match_opcode, 0 },
{"xor",         0, INSN_CLASS_I,   "d,s,j",  MATCH_XORI, MASK_XORI, match_opcode, INSN_ALIAS },
{"lwu",        64, INSN_CLASS_I, "d,o(s)",  MATCH_LWU, MASK_LWU, match_opcode, INSN_DREF|INSN_4_BYTE },
{"lwu",        64, INSN_CLASS_I, "d,A",  0, (int) M_LWU, match_never, INSN_MACRO },
{"ld",         64, INSN_CLASS_C, "d,Cn(Cc)",  MATCH_C_LDSP, MASK_C_LDSP, match_rd_nonzero, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"ld",         64, INSN_CLASS_C, "Ct,Cl(Cs)",  MATCH_C_LD, MASK_C_LD, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"ld",         64, INSN_CLASS_I, "d,o(s)", MATCH_LD, MASK_LD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"ld",         64, INSN_CLASS_I, "d,A",  0, (int) M_LD, match_never, INSN_MACRO },
{"sd",         64, INSN_CLASS_C, "CV,CN(Cc)",  MATCH_C_SDSP, MASK_C_SDSP, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"sd",         64, INSN_CLASS_C, "Ct,Cl(Cs)",  MATCH_C_SD, MASK_C_SD, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"sd",         64, INSN_CLASS_I, "t,q(s)",  MATCH_SD, MASK_SD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"sd",         64, INSN_CLASS_I, "t,A,s",  0, (int) M_SD, match_never, INSN_MACRO },
{"sext.w",     64, INSN_CLASS_C, "d,CU",  MATCH_C_ADDIW, MASK_C_ADDIW | MASK_RVC_IMM, match_rd_nonzero, INSN_ALIAS },
{"sext.w",     64, INSN_CLASS_I, "d,s",  MATCH_ADDIW, MASK_ADDIW | MASK_IMM, match_opcode, INSN_ALIAS },
{"addiw",      64, INSN_CLASS_C, "d,CU,Co",  MATCH_C_ADDIW, MASK_C_ADDIW, match_rd_nonzero, INSN_ALIAS },
{"addiw",      64, INSN_CLASS_I, "d,s,j",  MATCH_ADDIW, MASK_ADDIW, match_opcode, 0 },
{"addw",       64, INSN_CLASS_C, "Cs,Cw,Ct",  MATCH_C_ADDW, MASK_C_ADDW, match_opcode, INSN_ALIAS },
{"addw",       64, INSN_CLASS_C, "Cs,Ct,Cw",  MATCH_C_ADDW, MASK_C_ADDW, match_opcode, INSN_ALIAS },
{"addw",       64, INSN_CLASS_C, "d,CU,Co",  MATCH_C_ADDIW, MASK_C_ADDIW, match_rd_nonzero, INSN_ALIAS },
{"addw",       64, INSN_CLASS_I, "d,s,t",  MATCH_ADDW, MASK_ADDW, match_opcode, 0 },
{"addw",       64, INSN_CLASS_I, "d,s,j",  MATCH_ADDIW, MASK_ADDIW, match_opcode, INSN_ALIAS },
{"negw",       64, INSN_CLASS_I, "d,t",  MATCH_SUBW, MASK_SUBW | MASK_RS1, match_opcode, INSN_ALIAS }, /* sub 0 */
{"slliw",      64, INSN_CLASS_I, "d,s,<",   MATCH_SLLIW, MASK_SLLIW, match_opcode, 0 },
{"sllw",       64, INSN_CLASS_I, "d,s,t",   MATCH_SLLW, MASK_SLLW, match_opcode, 0 },
{"sllw",       64, INSN_CLASS_I, "d,s,<",   MATCH_SLLIW, MASK_SLLIW, match_opcode, INSN_ALIAS },
{"srliw",      64, INSN_CLASS_I, "d,s,<",   MATCH_SRLIW, MASK_SRLIW, match_opcode, 0 },
{"srlw",       64, INSN_CLASS_I, "d,s,t",   MATCH_SRLW, MASK_SRLW, match_opcode, 0 },
{"srlw",       64, INSN_CLASS_I, "d,s,<",   MATCH_SRLIW, MASK_SRLIW, match_opcode, INSN_ALIAS },
{"sraiw",      64, INSN_CLASS_I, "d,s,<",   MATCH_SRAIW, MASK_SRAIW, match_opcode, 0 },
{"sraw",       64, INSN_CLASS_I, "d,s,t",   MATCH_SRAW, MASK_SRAW, match_opcode, 0 },
{"sraw",       64, INSN_CLASS_I, "d,s,<",   MATCH_SRAIW, MASK_SRAIW, match_opcode, INSN_ALIAS },
{"subw",       64, INSN_CLASS_C, "Cs,Cw,Ct",  MATCH_C_SUBW, MASK_C_SUBW, match_opcode, INSN_ALIAS },
{"subw",       64, INSN_CLASS_I, "d,s,t",  MATCH_SUBW, MASK_SUBW, match_opcode, 0 },

/* Atomic memory operation instruction subset */
{"lr.w",         0, INSN_CLASS_A,   "d,0(s)",    MATCH_LR_W, MASK_LR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"sc.w",         0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_SC_W, MASK_SC_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoadd.w",     0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOADD_W, MASK_AMOADD_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoswap.w",    0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOSWAP_W, MASK_AMOSWAP_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoand.w",     0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOAND_W, MASK_AMOAND_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoor.w",      0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOOR_W, MASK_AMOOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoxor.w",     0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOXOR_W, MASK_AMOXOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomax.w",     0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAX_W, MASK_AMOMAX_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomaxu.w",    0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAXU_W, MASK_AMOMAXU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomin.w",     0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMIN_W, MASK_AMOMIN_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amominu.w",    0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMINU_W, MASK_AMOMINU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"lr.w.aq",      0, INSN_CLASS_A,   "d,0(s)",    MATCH_LR_W | MASK_AQ, MASK_LR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"sc.w.aq",      0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_SC_W | MASK_AQ, MASK_SC_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoadd.w.aq",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOADD_W | MASK_AQ, MASK_AMOADD_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoswap.w.aq", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOSWAP_W | MASK_AQ, MASK_AMOSWAP_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoand.w.aq",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOAND_W | MASK_AQ, MASK_AMOAND_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoor.w.aq",   0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOOR_W | MASK_AQ, MASK_AMOOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoxor.w.aq",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOXOR_W | MASK_AQ, MASK_AMOXOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomax.w.aq",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAX_W | MASK_AQ, MASK_AMOMAX_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomaxu.w.aq", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAXU_W | MASK_AQ, MASK_AMOMAXU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomin.w.aq",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMIN_W | MASK_AQ, MASK_AMOMIN_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amominu.w.aq", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMINU_W | MASK_AQ, MASK_AMOMINU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"lr.w.rl",      0, INSN_CLASS_A,   "d,0(s)",    MATCH_LR_W | MASK_RL, MASK_LR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"sc.w.rl",      0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_SC_W | MASK_RL, MASK_SC_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoadd.w.rl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOADD_W | MASK_RL, MASK_AMOADD_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoswap.w.rl", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOSWAP_W | MASK_RL, MASK_AMOSWAP_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoand.w.rl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOAND_W | MASK_RL, MASK_AMOAND_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoor.w.rl",   0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOOR_W | MASK_RL, MASK_AMOOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoxor.w.rl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOXOR_W | MASK_RL, MASK_AMOXOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomax.w.rl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAX_W | MASK_RL, MASK_AMOMAX_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomaxu.w.rl", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAXU_W | MASK_RL, MASK_AMOMAXU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomin.w.rl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMIN_W | MASK_RL, MASK_AMOMIN_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amominu.w.rl", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMINU_W | MASK_RL, MASK_AMOMINU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"lr.w.aqrl",    0, INSN_CLASS_A,   "d,0(s)",    MATCH_LR_W | MASK_AQRL, MASK_LR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"sc.w.aqrl",    0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_SC_W | MASK_AQRL, MASK_SC_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoadd.w.aqrl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOADD_W | MASK_AQRL, MASK_AMOADD_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoswap.w.aqrl", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOSWAP_W | MASK_AQRL, MASK_AMOSWAP_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoand.w.aqrl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOAND_W | MASK_AQRL, MASK_AMOAND_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoor.w.aqrl",   0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOOR_W | MASK_AQRL, MASK_AMOOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amoxor.w.aqrl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOXOR_W | MASK_AQRL, MASK_AMOXOR_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomax.w.aqrl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAX_W | MASK_AQRL, MASK_AMOMAX_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomaxu.w.aqrl", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMAXU_W | MASK_AQRL, MASK_AMOMAXU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amomin.w.aqrl",  0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMIN_W | MASK_AQRL, MASK_AMOMIN_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"amominu.w.aqrl", 0, INSN_CLASS_A,   "d,t,0(s)",  MATCH_AMOMINU_W | MASK_AQRL, MASK_AMOMINU_W | MASK_AQRL, match_opcode, INSN_DREF|INSN_4_BYTE },
{"lr.d",         64, INSN_CLASS_A , "d,0(s)",    MATCH_LR_D, MASK_LR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"sc.d",         64, INSN_CLASS_A , "d,t,0(s)",  MATCH_SC_D, MASK_SC_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoadd.d",     64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOADD_D, MASK_AMOADD_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoswap.d",    64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOSWAP_D, MASK_AMOSWAP_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoand.d",     64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOAND_D, MASK_AMOAND_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoor.d",      64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOOR_D, MASK_AMOOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoxor.d",     64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOXOR_D, MASK_AMOXOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomax.d",     64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAX_D, MASK_AMOMAX_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomaxu.d",    64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAXU_D, MASK_AMOMAXU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomin.d",     64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMIN_D, MASK_AMOMIN_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amominu.d",    64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMINU_D, MASK_AMOMINU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"lr.d.aq",      64, INSN_CLASS_A , "d,0(s)",    MATCH_LR_D | MASK_AQ, MASK_LR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"sc.d.aq",      64, INSN_CLASS_A , "d,t,0(s)",  MATCH_SC_D | MASK_AQ, MASK_SC_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoadd.d.aq",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOADD_D | MASK_AQ, MASK_AMOADD_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoswap.d.aq", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOSWAP_D | MASK_AQ, MASK_AMOSWAP_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoand.d.aq",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOAND_D | MASK_AQ, MASK_AMOAND_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoor.d.aq",   64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOOR_D | MASK_AQ, MASK_AMOOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoxor.d.aq",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOXOR_D | MASK_AQ, MASK_AMOXOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomax.d.aq",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAX_D | MASK_AQ, MASK_AMOMAX_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomaxu.d.aq", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAXU_D | MASK_AQ, MASK_AMOMAXU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomin.d.aq",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMIN_D | MASK_AQ, MASK_AMOMIN_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amominu.d.aq", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMINU_D | MASK_AQ, MASK_AMOMINU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"lr.d.rl",      64, INSN_CLASS_A , "d,0(s)",    MATCH_LR_D | MASK_RL, MASK_LR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"sc.d.rl",      64, INSN_CLASS_A , "d,t,0(s)",  MATCH_SC_D | MASK_RL, MASK_SC_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoadd.d.rl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOADD_D | MASK_RL, MASK_AMOADD_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoswap.d.rl", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOSWAP_D | MASK_RL, MASK_AMOSWAP_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoand.d.rl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOAND_D | MASK_RL, MASK_AMOAND_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoor.d.rl",   64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOOR_D | MASK_RL, MASK_AMOOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoxor.d.rl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOXOR_D | MASK_RL, MASK_AMOXOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomax.d.rl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAX_D | MASK_RL, MASK_AMOMAX_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomaxu.d.rl", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAXU_D | MASK_RL, MASK_AMOMAXU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomin.d.rl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMIN_D | MASK_RL, MASK_AMOMIN_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amominu.d.rl", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMINU_D | MASK_RL, MASK_AMOMINU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"lr.d.aqrl",    64, INSN_CLASS_A , "d,0(s)",    MATCH_LR_D | MASK_AQRL, MASK_LR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"sc.d.aqrl",    64, INSN_CLASS_A , "d,t,0(s)",  MATCH_SC_D | MASK_AQRL, MASK_SC_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoadd.d.aqrl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOADD_D | MASK_AQRL, MASK_AMOADD_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoswap.d.aqrl", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOSWAP_D | MASK_AQRL, MASK_AMOSWAP_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoand.d.aqrl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOAND_D | MASK_AQRL, MASK_AMOAND_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoor.d.aqrl",   64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOOR_D | MASK_AQRL, MASK_AMOOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amoxor.d.aqrl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOXOR_D | MASK_AQRL, MASK_AMOXOR_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomax.d.aqrl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAX_D | MASK_AQRL, MASK_AMOMAX_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomaxu.d.aqrl", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMAXU_D | MASK_AQRL, MASK_AMOMAXU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amomin.d.aqrl",  64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMIN_D | MASK_AQRL, MASK_AMOMIN_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },
{"amominu.d.aqrl", 64, INSN_CLASS_A , "d,t,0(s)",  MATCH_AMOMINU_D | MASK_AQRL, MASK_AMOMINU_D | MASK_AQRL, match_opcode, INSN_DREF|INSN_8_BYTE },

/* Multiply/Divide instruction subset */
{"mul",       0, INSN_CLASS_M,   "d,s,t",  MATCH_MUL, MASK_MUL, match_opcode, 0 },
{"mulh",      0, INSN_CLASS_M,   "d,s,t",  MATCH_MULH, MASK_MULH, match_opcode, 0 },
{"mulhu",     0, INSN_CLASS_M,   "d,s,t",  MATCH_MULHU, MASK_MULHU, match_opcode, 0 },
{"mulhsu",    0, INSN_CLASS_M,   "d,s,t",  MATCH_MULHSU, MASK_MULHSU, match_opcode, 0 },
{"div",       0, INSN_CLASS_M,   "d,s,t",  MATCH_DIV, MASK_DIV, match_opcode, 0 },
{"divu",      0, INSN_CLASS_M,   "d,s,t",  MATCH_DIVU, MASK_DIVU, match_opcode, 0 },
{"rem",       0, INSN_CLASS_M,   "d,s,t",  MATCH_REM, MASK_REM, match_opcode, 0 },
{"remu",      0, INSN_CLASS_M,   "d,s,t",  MATCH_REMU, MASK_REMU, match_opcode, 0 },
{"mulw",     64, INSN_CLASS_M, "d,s,t",  MATCH_MULW, MASK_MULW, match_opcode, 0 },
{"divw",     64, INSN_CLASS_M, "d,s,t",  MATCH_DIVW, MASK_DIVW, match_opcode, 0 },
{"divuw",    64, INSN_CLASS_M, "d,s,t",  MATCH_DIVUW, MASK_DIVUW, match_opcode, 0 },
{"remw",     64, INSN_CLASS_M, "d,s,t",  MATCH_REMW, MASK_REMW, match_opcode, 0 },
{"remuw",    64, INSN_CLASS_M, "d,s,t",  MATCH_REMUW, MASK_REMUW, match_opcode, 0 },

/* Single-precision floating-point instruction subset */
{"frcsr",     0, INSN_CLASS_F,   "d",  MATCH_FRCSR, MASK_FRCSR, match_opcode, INSN_ALIAS },
{"frcsr",     0, INSN_CLASS_ZFINX,   "d",  MATCH_FRCSR, MASK_FRCSR, match_opcode, INSN_ALIAS },
{"frsr",      0, INSN_CLASS_F,   "d",  MATCH_FRCSR, MASK_FRCSR, match_opcode, INSN_ALIAS },
{"frsr",      0, INSN_CLASS_ZFINX,   "d",  MATCH_FRCSR, MASK_FRCSR, match_opcode, INSN_ALIAS },
{"fscsr",     0, INSN_CLASS_F,   "s",  MATCH_FSCSR, MASK_FSCSR | MASK_RD, match_opcode, INSN_ALIAS },
{"fscsr",     0, INSN_CLASS_F,   "d,s",  MATCH_FSCSR, MASK_FSCSR, match_opcode, INSN_ALIAS },
{"fscsr",     0, INSN_CLASS_ZFINX,   "s",  MATCH_FSCSR, MASK_FSCSR | MASK_RD, match_opcode, INSN_ALIAS },
{"fscsr",     0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FSCSR, MASK_FSCSR, match_opcode, INSN_ALIAS },
{"fssr",      0, INSN_CLASS_F,   "s",  MATCH_FSCSR, MASK_FSCSR | MASK_RD, match_opcode, INSN_ALIAS },
{"fssr",      0, INSN_CLASS_F,   "d,s",  MATCH_FSCSR, MASK_FSCSR, match_opcode, INSN_ALIAS },
{"fssr",      0, INSN_CLASS_ZFINX,   "s",  MATCH_FSCSR, MASK_FSCSR | MASK_RD, match_opcode, INSN_ALIAS },
{"fssr",      0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FSCSR, MASK_FSCSR, match_opcode, INSN_ALIAS },
{"frrm",      0, INSN_CLASS_F,   "d",  MATCH_FRRM, MASK_FRRM, match_opcode, INSN_ALIAS },
{"frrm",      0, INSN_CLASS_ZFINX,   "d",  MATCH_FRRM, MASK_FRRM, match_opcode, INSN_ALIAS },
{"fsrm",      0, INSN_CLASS_F,   "s",  MATCH_FSRM, MASK_FSRM | MASK_RD, match_opcode, INSN_ALIAS },
{"fsrm",      0, INSN_CLASS_F,   "d,s",  MATCH_FSRM, MASK_FSRM, match_opcode, INSN_ALIAS },
{"fsrm",      0, INSN_CLASS_ZFINX,   "s",  MATCH_FSRM, MASK_FSRM | MASK_RD, match_opcode, INSN_ALIAS },
{"fsrm",      0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FSRM, MASK_FSRM, match_opcode, INSN_ALIAS },
{"fsrmi",     0, INSN_CLASS_F,   "d,Z",  MATCH_FSRMI, MASK_FSRMI, match_opcode, INSN_ALIAS },
{"fsrmi",     0, INSN_CLASS_F,   "Z",  MATCH_FSRMI, MASK_FSRMI | MASK_RD, match_opcode, INSN_ALIAS },
{"fsrmi",     0, INSN_CLASS_ZFINX,   "d,Z",  MATCH_FSRMI, MASK_FSRMI, match_opcode, INSN_ALIAS },
{"fsrmi",     0, INSN_CLASS_ZFINX,   "Z",  MATCH_FSRMI, MASK_FSRMI | MASK_RD, match_opcode, INSN_ALIAS },
{"frflags",   0, INSN_CLASS_F,   "d",  MATCH_FRFLAGS, MASK_FRFLAGS, match_opcode, INSN_ALIAS },
{"frflags",   0, INSN_CLASS_ZFINX,   "d",  MATCH_FRFLAGS, MASK_FRFLAGS, match_opcode, INSN_ALIAS },
{"fsflags",   0, INSN_CLASS_F,   "s",  MATCH_FSFLAGS, MASK_FSFLAGS | MASK_RD, match_opcode, INSN_ALIAS },
{"fsflags",   0, INSN_CLASS_F,   "d,s",  MATCH_FSFLAGS, MASK_FSFLAGS, match_opcode, INSN_ALIAS },
{"fsflags",   0, INSN_CLASS_ZFINX,   "s",  MATCH_FSFLAGS, MASK_FSFLAGS | MASK_RD, match_opcode, INSN_ALIAS },
{"fsflags",   0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FSFLAGS, MASK_FSFLAGS, match_opcode, INSN_ALIAS },
{"fsflagsi",  0, INSN_CLASS_F,   "d,Z",  MATCH_FSFLAGSI, MASK_FSFLAGSI, match_opcode, INSN_ALIAS },
{"fsflagsi",  0, INSN_CLASS_F,   "Z",  MATCH_FSFLAGSI, MASK_FSFLAGSI | MASK_RD, match_opcode, INSN_ALIAS },
{"fsflagsi",  0, INSN_CLASS_ZFINX,   "d,Z",  MATCH_FSFLAGSI, MASK_FSFLAGSI, match_opcode, INSN_ALIAS },
{"fsflagsi",  0, INSN_CLASS_ZFINX,   "Z",  MATCH_FSFLAGSI, MASK_FSFLAGSI | MASK_RD, match_opcode, INSN_ALIAS },
{"flw",      32, INSN_CLASS_F_AND_C, "D,Cm(Cc)",  MATCH_C_FLWSP, MASK_C_FLWSP, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"flw",      32, INSN_CLASS_F_AND_C, "CD,Ck(Cs)",  MATCH_C_FLW, MASK_C_FLW, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"flw",       0, INSN_CLASS_F,   "D,o(s)",  MATCH_FLW, MASK_FLW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"flw",       0, INSN_CLASS_F,   "D,A,s",  0, (int) M_FLW, match_never, INSN_MACRO },
{"fsw",      32, INSN_CLASS_F_AND_C, "CT,CM(Cc)",  MATCH_C_FSWSP, MASK_C_FSWSP, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"fsw",      32, INSN_CLASS_F_AND_C, "CD,Ck(Cs)",  MATCH_C_FSW, MASK_C_FSW, match_opcode, INSN_ALIAS|INSN_DREF|INSN_4_BYTE },
{"fsw",       0, INSN_CLASS_F,   "T,q(s)",  MATCH_FSW, MASK_FSW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"fsw",       0, INSN_CLASS_F,   "T,A,s",  0, (int) M_FSW, match_never, INSN_MACRO },

{"fmv.x.w",    0, INSN_CLASS_F,   "d,S",  MATCH_FMV_X_S, MASK_FMV_X_S, match_opcode, 0 },
{"fmv.w.x",    0, INSN_CLASS_F,   "D,s",  MATCH_FMV_S_X, MASK_FMV_S_X, match_opcode, 0 },

{"fmv.x.s",    0, INSN_CLASS_F,   "d,S",  MATCH_FMV_X_S, MASK_FMV_X_S, match_opcode, 0 },
{"fmv.s.x",    0, INSN_CLASS_F,   "D,s",  MATCH_FMV_S_X, MASK_FMV_S_X, match_opcode, 0 },

{"fmv.s",      0, INSN_CLASS_F,   "D,U",  MATCH_FSGNJ_S, MASK_FSGNJ_S, match_rs1_eq_rs2, INSN_ALIAS },
{"fneg.s",     0, INSN_CLASS_F,   "D,U",  MATCH_FSGNJN_S, MASK_FSGNJN_S, match_rs1_eq_rs2, INSN_ALIAS },
{"fneg.s",     0, INSN_CLASS_ZFINX,   "d,x",  MATCH_FSGNJN_S, MASK_FSGNJN_S, match_rs1_eq_rs2, INSN_ALIAS },
{"fabs.s",     0, INSN_CLASS_F,   "D,U",  MATCH_FSGNJX_S, MASK_FSGNJX_S, match_rs1_eq_rs2, INSN_ALIAS },
{"fabs.s",     0, INSN_CLASS_ZFINX,   "d,x",  MATCH_FSGNJX_S, MASK_FSGNJX_S, match_rs1_eq_rs2, INSN_ALIAS },
{"fsgnj.s",    0, INSN_CLASS_F,   "D,S,T",  MATCH_FSGNJ_S, MASK_FSGNJ_S, match_opcode, 0 },
{"fsgnj.s",    0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FSGNJ_S, MASK_FSGNJ_S, match_opcode, 0 },
{"fsgnjn.s",   0, INSN_CLASS_F,   "D,S,T",  MATCH_FSGNJN_S, MASK_FSGNJN_S, match_opcode, 0 },
{"fsgnjn.s",   0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FSGNJN_S, MASK_FSGNJN_S, match_opcode, 0 },
{"fsgnjx.s",   0, INSN_CLASS_F,   "D,S,T",  MATCH_FSGNJX_S, MASK_FSGNJX_S, match_opcode, 0 },
{"fsgnjx.s",   0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FSGNJX_S, MASK_FSGNJX_S, match_opcode, 0 },
{"fadd.s",     0, INSN_CLASS_F,   "D,S,T",  MATCH_FADD_S | MASK_RM, MASK_FADD_S | MASK_RM, match_opcode, 0 },
{"fadd.s",     0, INSN_CLASS_F,   "D,S,T,m",  MATCH_FADD_S, MASK_FADD_S, match_opcode, 0 },
{"fadd.s",     0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FADD_S | MASK_RM, MASK_FADD_S | MASK_RM, match_opcode, 0 },
{"fadd.s",     0, INSN_CLASS_ZFINX,   "d,s,t,m",  MATCH_FADD_S, MASK_FADD_S, match_opcode, 0 },
{"fsub.s",     0, INSN_CLASS_F,   "D,S,T",  MATCH_FSUB_S | MASK_RM, MASK_FSUB_S | MASK_RM, match_opcode, 0 },
{"fsub.s",     0, INSN_CLASS_F,   "D,S,T,m",  MATCH_FSUB_S, MASK_FSUB_S, match_opcode, 0 },
{"fsub.s",     0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FSUB_S | MASK_RM, MASK_FSUB_S | MASK_RM, match_opcode, 0 },
{"fsub.s",     0, INSN_CLASS_ZFINX,   "d,s,t,m",  MATCH_FSUB_S, MASK_FSUB_S, match_opcode, 0 },
{"fmul.s",     0, INSN_CLASS_F,   "D,S,T",  MATCH_FMUL_S | MASK_RM, MASK_FMUL_S | MASK_RM, match_opcode, 0 },
{"fmul.s",     0, INSN_CLASS_F,   "D,S,T,m",  MATCH_FMUL_S, MASK_FMUL_S, match_opcode, 0 },
{"fmul.s",     0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FMUL_S | MASK_RM, MASK_FMUL_S | MASK_RM, match_opcode, 0 },
{"fmul.s",     0, INSN_CLASS_ZFINX,   "d,s,t,m",  MATCH_FMUL_S, MASK_FMUL_S, match_opcode, 0 },
{"fdiv.s",     0, INSN_CLASS_F,   "D,S,T",  MATCH_FDIV_S | MASK_RM, MASK_FDIV_S | MASK_RM, match_opcode, 0 },
{"fdiv.s",     0, INSN_CLASS_F,   "D,S,T,m",  MATCH_FDIV_S, MASK_FDIV_S, match_opcode, 0 },
{"fdiv.s",     0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FDIV_S | MASK_RM, MASK_FDIV_S | MASK_RM, match_opcode, 0 },
{"fdiv.s",     0, INSN_CLASS_ZFINX,   "d,s,t,m",  MATCH_FDIV_S, MASK_FDIV_S, match_opcode, 0 },
{"fsqrt.s",    0, INSN_CLASS_F,   "D,S",  MATCH_FSQRT_S | MASK_RM, MASK_FSQRT_S | MASK_RM, match_opcode, 0 },
{"fsqrt.s",    0, INSN_CLASS_F,   "D,S,m",  MATCH_FSQRT_S, MASK_FSQRT_S, match_opcode, 0 },
{"fsqrt.s",    0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FSQRT_S | MASK_RM, MASK_FSQRT_S | MASK_RM, match_opcode, 0 },
{"fsqrt.s",    0, INSN_CLASS_ZFINX,   "d,s,m",  MATCH_FSQRT_S, MASK_FSQRT_S, match_opcode, 0 },
{"fmin.s",     0, INSN_CLASS_F,   "D,S,T",  MATCH_FMIN_S, MASK_FMIN_S, match_opcode, 0 },
{"fmin.s",     0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FMIN_S, MASK_FMIN_S, match_opcode, 0 },
{"fmax.s",     0, INSN_CLASS_F,   "D,S,T",  MATCH_FMAX_S, MASK_FMAX_S, match_opcode, 0 },
{"fmax.s",     0, INSN_CLASS_ZFINX,   "d,s,t",  MATCH_FMAX_S, MASK_FMAX_S, match_opcode, 0 },
{"fmadd.s",    0, INSN_CLASS_F,   "D,S,T,R",  MATCH_FMADD_S | MASK_RM, MASK_FMADD_S | MASK_RM, match_opcode, 0 },
{"fmadd.s",    0, INSN_CLASS_F,   "D,S,T,R,m",  MATCH_FMADD_S, MASK_FMADD_S, match_opcode, 0 },
{"fmadd.s",    0, INSN_CLASS_ZFINX,   "d,s,t,r",  MATCH_FMADD_S | MASK_RM, MASK_FMADD_S | MASK_RM, match_opcode, 0 },
{"fmadd.s",    0, INSN_CLASS_ZFINX,   "d,s,t,r,m",  MATCH_FMADD_S, MASK_FMADD_S, match_opcode, 0 },
{"fnmadd.s",   0, INSN_CLASS_F,   "D,S,T,R",  MATCH_FNMADD_S | MASK_RM, MASK_FNMADD_S | MASK_RM, match_opcode, 0 },
{"fnmadd.s",   0, INSN_CLASS_F,   "D,S,T,R,m",  MATCH_FNMADD_S, MASK_FNMADD_S, match_opcode, 0 },
{"fnmadd.s",   0, INSN_CLASS_ZFINX,   "d,s,t,r",  MATCH_FNMADD_S | MASK_RM, MASK_FNMADD_S | MASK_RM, match_opcode, 0 },
{"fnmadd.s",   0, INSN_CLASS_ZFINX,   "d,s,t,r,m",  MATCH_FNMADD_S, MASK_FNMADD_S, match_opcode, 0 },
{"fmsub.s",    0, INSN_CLASS_F,   "D,S,T,R",  MATCH_FMSUB_S | MASK_RM, MASK_FMSUB_S | MASK_RM, match_opcode, 0 },
{"fmsub.s",    0, INSN_CLASS_F,   "D,S,T,R,m",  MATCH_FMSUB_S, MASK_FMSUB_S, match_opcode, 0 },
{"fmsub.s",    0, INSN_CLASS_ZFINX,   "d,s,t,r",  MATCH_FMSUB_S | MASK_RM, MASK_FMSUB_S | MASK_RM, match_opcode, 0 },
{"fmsub.s",    0, INSN_CLASS_ZFINX,   "d,s,t,r,m",  MATCH_FMSUB_S, MASK_FMSUB_S, match_opcode, 0 },
{"fnmsub.s",   0, INSN_CLASS_F,   "D,S,T,R",  MATCH_FNMSUB_S | MASK_RM, MASK_FNMSUB_S | MASK_RM, match_opcode, 0 },
{"fnmsub.s",   0, INSN_CLASS_F,   "D,S,T,R,m",  MATCH_FNMSUB_S, MASK_FNMSUB_S, match_opcode, 0 },
{"fnmsub.s",   0, INSN_CLASS_ZFINX,   "d,s,t,r",  MATCH_FNMSUB_S | MASK_RM, MASK_FNMSUB_S | MASK_RM, match_opcode, 0 },
{"fnmsub.s",   0, INSN_CLASS_ZFINX,   "d,s,t,r,m",  MATCH_FNMSUB_S, MASK_FNMSUB_S, match_opcode, 0 },
{"fcvt.w.s",   0, INSN_CLASS_F,   "d,S",  MATCH_FCVT_W_S | MASK_RM, MASK_FCVT_W_S | MASK_RM, match_opcode, 0 },
{"fcvt.w.s",   0, INSN_CLASS_F,   "d,S,m",  MATCH_FCVT_W_S, MASK_FCVT_W_S, match_opcode, 0 },
{"fcvt.w.s",   0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FCVT_W_S | MASK_RM, MASK_FCVT_W_S | MASK_RM, match_opcode, 0 },
{"fcvt.w.s",   0, INSN_CLASS_ZFINX,   "d,s,m",  MATCH_FCVT_W_S, MASK_FCVT_W_S, match_opcode, 0 },
{"fcvt.wu.s",  0, INSN_CLASS_F,   "d,S",  MATCH_FCVT_WU_S | MASK_RM, MASK_FCVT_WU_S | MASK_RM, match_opcode, 0 },
{"fcvt.wu.s",  0, INSN_CLASS_F,   "d,S,m",  MATCH_FCVT_WU_S, MASK_FCVT_WU_S, match_opcode, 0 },
{"fcvt.wu.s",  0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FCVT_WU_S | MASK_RM, MASK_FCVT_WU_S | MASK_RM, match_opcode, 0 },
{"fcvt.wu.s",  0, INSN_CLASS_ZFINX,   "d,s,m",  MATCH_FCVT_WU_S, MASK_FCVT_WU_S, match_opcode, 0 },
{"fcvt.s.w",   0, INSN_CLASS_F,   "D,s",  MATCH_FCVT_S_W | MASK_RM, MASK_FCVT_S_W | MASK_RM, match_opcode, 0 },
{"fcvt.s.w",   0, INSN_CLASS_F,   "D,s,m",  MATCH_FCVT_S_W, MASK_FCVT_S_W, match_opcode, 0 },
{"fcvt.s.w",   0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FCVT_S_W | MASK_RM, MASK_FCVT_S_W | MASK_RM, match_opcode, 0 },
{"fcvt.s.w",   0, INSN_CLASS_ZFINX,   "d,s,m",  MATCH_FCVT_S_W, MASK_FCVT_S_W, match_opcode, 0 },
{"fcvt.s.wu",  0, INSN_CLASS_F,   "D,s",  MATCH_FCVT_S_WU | MASK_RM, MASK_FCVT_S_W | MASK_RM, match_opcode, 0 },
{"fcvt.s.wu",  0, INSN_CLASS_F,   "D,s,m",  MATCH_FCVT_S_WU, MASK_FCVT_S_WU, match_opcode, 0 },
{"fcvt.s.wu",  0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FCVT_S_WU | MASK_RM, MASK_FCVT_S_W | MASK_RM, match_opcode, 0 },
{"fcvt.s.wu",  0, INSN_CLASS_ZFINX,   "d,s,m",  MATCH_FCVT_S_WU, MASK_FCVT_S_WU, match_opcode, 0 },
{"fclass.s",   0, INSN_CLASS_F,   "d,S",  MATCH_FCLASS_S, MASK_FCLASS_S, match_opcode, 0 },
{"fclass.s",   0, INSN_CLASS_ZFINX,   "d,s",  MATCH_FCLASS_S, MASK_FCLASS_S, match_opcode, 0 },
{"feq.s",      0, INSN_CLASS_F,   "d,S,T",    MATCH_FEQ_S, MASK_FEQ_S, match_opcode, 0 },
{"feq.s",      0, INSN_CLASS_ZFINX,   "d,s,t",    MATCH_FEQ_S, MASK_FEQ_S, match_opcode, 0 },
{"flt.s",      0, INSN_CLASS_F,   "d,S,T",    MATCH_FLT_S, MASK_FLT_S, match_opcode, 0 },
{"flt.s",      0, INSN_CLASS_ZFINX,   "d,s,t",    MATCH_FLT_S, MASK_FLT_S, match_opcode, 0 },
{"fle.s",      0, INSN_CLASS_F,   "d,S,T",    MATCH_FLE_S, MASK_FLE_S, match_opcode, 0 },
{"fle.s",      0, INSN_CLASS_ZFINX,   "d,s,t",    MATCH_FLE_S, MASK_FLE_S, match_opcode, 0 },
{"fgt.s",      0, INSN_CLASS_F,   "d,T,S",    MATCH_FLT_S, MASK_FLT_S, match_opcode, 0 },
{"fgt.s",      0, INSN_CLASS_ZFINX,   "d,t,s",    MATCH_FLT_S, MASK_FLT_S, match_opcode, 0 },
{"fge.s",      0, INSN_CLASS_F,   "d,T,S",    MATCH_FLE_S, MASK_FLE_S, match_opcode, 0 },
{"fge.s",      0, INSN_CLASS_ZFINX,   "d,t,s",    MATCH_FLE_S, MASK_FLE_S, match_opcode, 0 },
{"fcvt.l.s",  64, INSN_CLASS_F, "d,S",  MATCH_FCVT_L_S | MASK_RM, MASK_FCVT_L_S | MASK_RM, match_opcode, 0 },
{"fcvt.l.s",  64, INSN_CLASS_F, "d,S,m",  MATCH_FCVT_L_S, MASK_FCVT_L_S, match_opcode, 0 },
{"fcvt.l.s",  64, INSN_CLASS_ZFINX, "d,s",  MATCH_FCVT_L_S | MASK_RM, MASK_FCVT_L_S | MASK_RM, match_opcode, 0 },
{"fcvt.l.s",  64, INSN_CLASS_ZFINX, "d,s,m",  MATCH_FCVT_L_S, MASK_FCVT_L_S, match_opcode, 0 },
{"fcvt.lu.s", 64, INSN_CLASS_F, "d,S",  MATCH_FCVT_LU_S | MASK_RM, MASK_FCVT_LU_S | MASK_RM, match_opcode, 0 },
{"fcvt.lu.s", 64, INSN_CLASS_F, "d,S,m",  MATCH_FCVT_LU_S, MASK_FCVT_LU_S, match_opcode, 0 },
{"fcvt.lu.s", 64, INSN_CLASS_ZFINX, "d,s",  MATCH_FCVT_LU_S | MASK_RM, MASK_FCVT_LU_S | MASK_RM, match_opcode, 0 },
{"fcvt.lu.s", 64, INSN_CLASS_ZFINX, "d,s,m",  MATCH_FCVT_LU_S, MASK_FCVT_LU_S, match_opcode, 0 },
{"fcvt.s.l",  64, INSN_CLASS_F, "D,s",  MATCH_FCVT_S_L | MASK_RM, MASK_FCVT_S_L | MASK_RM, match_opcode, 0 },
{"fcvt.s.l",  64, INSN_CLASS_F, "D,s,m",  MATCH_FCVT_S_L, MASK_FCVT_S_L, match_opcode, 0 },
{"fcvt.s.l",  64, INSN_CLASS_ZFINX, "d,s",  MATCH_FCVT_S_L | MASK_RM, MASK_FCVT_S_L | MASK_RM, match_opcode, 0 },
{"fcvt.s.l",  64, INSN_CLASS_ZFINX, "d,s,m",  MATCH_FCVT_S_L, MASK_FCVT_S_L, match_opcode, 0 },
{"fcvt.s.lu", 64, INSN_CLASS_F, "D,s",  MATCH_FCVT_S_LU | MASK_RM, MASK_FCVT_S_L | MASK_RM, match_opcode, 0 },
{"fcvt.s.lu", 64, INSN_CLASS_F, "D,s,m",  MATCH_FCVT_S_LU, MASK_FCVT_S_LU, match_opcode, 0 },
{"fcvt.s.lu", 64, INSN_CLASS_ZFINX, "d,s",  MATCH_FCVT_S_LU | MASK_RM, MASK_FCVT_S_L | MASK_RM, match_opcode, 0 },
{"fcvt.s.lu", 64, INSN_CLASS_ZFINX, "d,s,m",  MATCH_FCVT_S_LU, MASK_FCVT_S_LU, match_opcode, 0 },

/* Double-precision floating-point instruction subset */
{"fld",        0, INSN_CLASS_D_AND_C,   "D,Cn(Cc)",  MATCH_C_FLDSP, MASK_C_FLDSP, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"fld",        0, INSN_CLASS_D_AND_C,   "CD,Cl(Cs)",  MATCH_C_FLD, MASK_C_FLD, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"fld",        0, INSN_CLASS_D,   "D,o(s)",  MATCH_FLD, MASK_FLD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"fld",        0, INSN_CLASS_D,   "D,A,s",  0, (int) M_FLD, match_never, INSN_MACRO },
{"fsd",        0, INSN_CLASS_D_AND_C,   "CT,CN(Cc)",  MATCH_C_FSDSP, MASK_C_FSDSP, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"fsd",        0, INSN_CLASS_D_AND_C,   "CD,Cl(Cs)",  MATCH_C_FSD, MASK_C_FSD, match_opcode, INSN_ALIAS|INSN_DREF|INSN_8_BYTE },
{"fsd",        0, INSN_CLASS_D,   "T,q(s)",  MATCH_FSD, MASK_FSD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"fsd",        0, INSN_CLASS_D,   "T,A,s",  0, (int) M_FSD, match_never, INSN_MACRO },
{"fmv.d",      0, INSN_CLASS_D,   "D,U",  MATCH_FSGNJ_D, MASK_FSGNJ_D, match_rs1_eq_rs2, INSN_ALIAS },
{"fneg.d",     0, INSN_CLASS_D,   "D,U",  MATCH_FSGNJN_D, MASK_FSGNJN_D, match_rs1_eq_rs2, INSN_ALIAS },
{"fneg.d",     0, INSN_CLASS_ZDINX,   "d,x",  MATCH_FSGNJN_D, MASK_FSGNJN_D, match_rs1_eq_rs2, INSN_ALIAS },
{"fabs.d",     0, INSN_CLASS_D,   "D,U",  MATCH_FSGNJX_D, MASK_FSGNJX_D, match_rs1_eq_rs2, INSN_ALIAS },
{"fabs.d",     0, INSN_CLASS_ZDINX,   "d,x",  MATCH_FSGNJX_D, MASK_FSGNJX_D, match_rs1_eq_rs2, INSN_ALIAS },
{"fsgnj.d",    0, INSN_CLASS_D,   "D,S,T",  MATCH_FSGNJ_D, MASK_FSGNJ_D, match_opcode, 0 },
{"fsgnj.d",    0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FSGNJ_D, MASK_FSGNJ_D, match_opcode, 0 },
{"fsgnjn.d",   0, INSN_CLASS_D,   "D,S,T",  MATCH_FSGNJN_D, MASK_FSGNJN_D, match_opcode, 0 },
{"fsgnjn.d",   0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FSGNJN_D, MASK_FSGNJN_D, match_opcode, 0 },
{"fsgnjx.d",   0, INSN_CLASS_D,   "D,S,T",  MATCH_FSGNJX_D, MASK_FSGNJX_D, match_opcode, 0 },
{"fsgnjx.d",   0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FSGNJX_D, MASK_FSGNJX_D, match_opcode, 0 },
{"fadd.d",     0, INSN_CLASS_D,   "D,S,T",  MATCH_FADD_D | MASK_RM, MASK_FADD_D | MASK_RM, match_opcode, 0 },
{"fadd.d",     0, INSN_CLASS_D,   "D,S,T,m",  MATCH_FADD_D, MASK_FADD_D, match_opcode, 0 },
{"fadd.d",     0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FADD_D | MASK_RM, MASK_FADD_D | MASK_RM, match_opcode, 0 },
{"fadd.d",     0, INSN_CLASS_ZDINX,   "d,s,t,m",  MATCH_FADD_D, MASK_FADD_D, match_opcode, 0 },
{"fsub.d",     0, INSN_CLASS_D,   "D,S,T",  MATCH_FSUB_D | MASK_RM, MASK_FSUB_D | MASK_RM, match_opcode, 0 },
{"fsub.d",     0, INSN_CLASS_D,   "D,S,T,m",  MATCH_FSUB_D, MASK_FSUB_D, match_opcode, 0 },
{"fsub.d",     0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FSUB_D | MASK_RM, MASK_FSUB_D | MASK_RM, match_opcode, 0 },
{"fsub.d",     0, INSN_CLASS_ZDINX,   "d,s,t,m",  MATCH_FSUB_D, MASK_FSUB_D, match_opcode, 0 },
{"fmul.d",     0, INSN_CLASS_D,   "D,S,T",  MATCH_FMUL_D | MASK_RM, MASK_FMUL_D | MASK_RM, match_opcode, 0 },
{"fmul.d",     0, INSN_CLASS_D,   "D,S,T,m",  MATCH_FMUL_D, MASK_FMUL_D, match_opcode, 0 },
{"fmul.d",     0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FMUL_D | MASK_RM, MASK_FMUL_D | MASK_RM, match_opcode, 0 },
{"fmul.d",     0, INSN_CLASS_ZDINX,   "d,s,t,m",  MATCH_FMUL_D, MASK_FMUL_D, match_opcode, 0 },
{"fdiv.d",     0, INSN_CLASS_D,   "D,S,T",  MATCH_FDIV_D | MASK_RM, MASK_FDIV_D | MASK_RM, match_opcode, 0 },
{"fdiv.d",     0, INSN_CLASS_D,   "D,S,T,m",  MATCH_FDIV_D, MASK_FDIV_D, match_opcode, 0 },
{"fdiv.d",     0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FDIV_D | MASK_RM, MASK_FDIV_D | MASK_RM, match_opcode, 0 },
{"fdiv.d",     0, INSN_CLASS_ZDINX,   "d,s,t,m",  MATCH_FDIV_D, MASK_FDIV_D, match_opcode, 0 },
{"fsqrt.d",    0, INSN_CLASS_D,   "D,S",  MATCH_FSQRT_D | MASK_RM, MASK_FSQRT_D | MASK_RM, match_opcode, 0 },
{"fsqrt.d",    0, INSN_CLASS_D,   "D,S,m",  MATCH_FSQRT_D, MASK_FSQRT_D, match_opcode, 0 },
{"fsqrt.d",    0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FSQRT_D | MASK_RM, MASK_FSQRT_D | MASK_RM, match_opcode, 0 },
{"fsqrt.d",    0, INSN_CLASS_ZDINX,   "d,s,m",  MATCH_FSQRT_D, MASK_FSQRT_D, match_opcode, 0 },
{"fmin.d",     0, INSN_CLASS_D,   "D,S,T",  MATCH_FMIN_D, MASK_FMIN_D, match_opcode, 0 },
{"fmin.d",     0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FMIN_D, MASK_FMIN_D, match_opcode, 0 },
{"fmax.d",     0, INSN_CLASS_D,   "D,S,T",  MATCH_FMAX_D, MASK_FMAX_D, match_opcode, 0 },
{"fmax.d",     0, INSN_CLASS_ZDINX,   "d,s,t",  MATCH_FMAX_D, MASK_FMAX_D, match_opcode, 0 },
{"fmadd.d",    0, INSN_CLASS_D,   "D,S,T,R",  MATCH_FMADD_D | MASK_RM, MASK_FMADD_D | MASK_RM, match_opcode, 0 },
{"fmadd.d",    0, INSN_CLASS_D,   "D,S,T,R,m",  MATCH_FMADD_D, MASK_FMADD_D, match_opcode, 0 },
{"fmadd.d",    0, INSN_CLASS_ZDINX,   "d,s,t,r",  MATCH_FMADD_D | MASK_RM, MASK_FMADD_D | MASK_RM, match_opcode, 0 },
{"fmadd.d",    0, INSN_CLASS_ZDINX,   "d,s,t,r,m",  MATCH_FMADD_D, MASK_FMADD_D, match_opcode, 0 },
{"fnmadd.d",   0, INSN_CLASS_D,   "D,S,T,R",  MATCH_FNMADD_D | MASK_RM, MASK_FNMADD_D | MASK_RM, match_opcode, 0 },
{"fnmadd.d",   0, INSN_CLASS_D,   "D,S,T,R,m",  MATCH_FNMADD_D, MASK_FNMADD_D, match_opcode, 0 },
{"fnmadd.d",   0, INSN_CLASS_ZDINX,   "d,s,t,r",  MATCH_FNMADD_D | MASK_RM, MASK_FNMADD_D | MASK_RM, match_opcode, 0 },
{"fnmadd.d",   0, INSN_CLASS_ZDINX,   "d,s,t,r,m",  MATCH_FNMADD_D, MASK_FNMADD_D, match_opcode, 0 },
{"fmsub.d",    0, INSN_CLASS_D,   "D,S,T,R",  MATCH_FMSUB_D | MASK_RM, MASK_FMSUB_D | MASK_RM, match_opcode, 0 },
{"fmsub.d",    0, INSN_CLASS_D,   "D,S,T,R,m",  MATCH_FMSUB_D, MASK_FMSUB_D, match_opcode, 0 },
{"fmsub.d",    0, INSN_CLASS_ZDINX,   "d,s,t,r",  MATCH_FMSUB_D | MASK_RM, MASK_FMSUB_D | MASK_RM, match_opcode, 0 },
{"fmsub.d",    0, INSN_CLASS_ZDINX,   "d,s,t,r,m",  MATCH_FMSUB_D, MASK_FMSUB_D, match_opcode, 0 },
{"fnmsub.d",   0, INSN_CLASS_D,   "D,S,T,R",  MATCH_FNMSUB_D | MASK_RM, MASK_FNMSUB_D | MASK_RM, match_opcode, 0 },
{"fnmsub.d",   0, INSN_CLASS_D,   "D,S,T,R,m",  MATCH_FNMSUB_D, MASK_FNMSUB_D, match_opcode, 0 },
{"fnmsub.d",   0, INSN_CLASS_ZDINX,   "d,s,t,r",  MATCH_FNMSUB_D | MASK_RM, MASK_FNMSUB_D | MASK_RM, match_opcode, 0 },
{"fnmsub.d",   0, INSN_CLASS_ZDINX,   "d,s,t,r,m",  MATCH_FNMSUB_D, MASK_FNMSUB_D, match_opcode, 0 },
{"fcvt.w.d",   0, INSN_CLASS_D,   "d,S",  MATCH_FCVT_W_D | MASK_RM, MASK_FCVT_W_D | MASK_RM, match_opcode, 0 },
{"fcvt.w.d",   0, INSN_CLASS_D,   "d,S,m",  MATCH_FCVT_W_D, MASK_FCVT_W_D, match_opcode, 0 },
{"fcvt.w.d",   0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCVT_W_D | MASK_RM, MASK_FCVT_W_D | MASK_RM, match_opcode, 0 },
{"fcvt.w.d",   0, INSN_CLASS_ZDINX,   "d,s,m",  MATCH_FCVT_W_D, MASK_FCVT_W_D, match_opcode, 0 },
{"fcvt.wu.d",  0, INSN_CLASS_D,   "d,S",  MATCH_FCVT_WU_D | MASK_RM, MASK_FCVT_WU_D | MASK_RM, match_opcode, 0 },
{"fcvt.wu.d",  0, INSN_CLASS_D,   "d,S,m",  MATCH_FCVT_WU_D, MASK_FCVT_WU_D, match_opcode, 0 },
{"fcvt.wu.d",  0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCVT_WU_D | MASK_RM, MASK_FCVT_WU_D | MASK_RM, match_opcode, 0 },
{"fcvt.wu.d",  0, INSN_CLASS_ZDINX,   "d,s,m",  MATCH_FCVT_WU_D, MASK_FCVT_WU_D, match_opcode, 0 },
{"fcvt.d.w",   0, INSN_CLASS_D,   "D,s",  MATCH_FCVT_D_W, MASK_FCVT_D_W | MASK_RM, match_opcode, 0 },
{"fcvt.d.w",   0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCVT_D_W, MASK_FCVT_D_W | MASK_RM, match_opcode, 0 },
{"fcvt.d.wu",  0, INSN_CLASS_D,   "D,s",  MATCH_FCVT_D_WU, MASK_FCVT_D_WU | MASK_RM, match_opcode, 0 },
{"fcvt.d.wu",  0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCVT_D_WU, MASK_FCVT_D_WU | MASK_RM, match_opcode, 0 },
{"fcvt.d.s",   0, INSN_CLASS_D,   "D,S",  MATCH_FCVT_D_S, MASK_FCVT_D_S | MASK_RM, match_opcode, 0 },
{"fcvt.d.s",   0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCVT_D_S, MASK_FCVT_D_S | MASK_RM, match_opcode, 0 },
{"fcvt.s.d",   0, INSN_CLASS_D,   "D,S",  MATCH_FCVT_S_D | MASK_RM, MASK_FCVT_S_D | MASK_RM, match_opcode, 0 },
{"fcvt.s.d",   0, INSN_CLASS_D,   "D,S,m",  MATCH_FCVT_S_D, MASK_FCVT_S_D, match_opcode, 0 },
{"fcvt.s.d",   0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCVT_S_D | MASK_RM, MASK_FCVT_S_D | MASK_RM, match_opcode, 0 },
{"fcvt.s.d",   0, INSN_CLASS_ZDINX,   "d,s,m",  MATCH_FCVT_S_D, MASK_FCVT_S_D, match_opcode, 0 },
{"fclass.d",   0, INSN_CLASS_D,   "d,S",  MATCH_FCLASS_D, MASK_FCLASS_D, match_opcode, 0 },
{"fclass.d",   0, INSN_CLASS_ZDINX,   "d,s",  MATCH_FCLASS_D, MASK_FCLASS_D, match_opcode, 0 },
{"feq.d",      0, INSN_CLASS_D,   "d,S,T",    MATCH_FEQ_D, MASK_FEQ_D, match_opcode, 0 },
{"feq.d",      0, INSN_CLASS_ZDINX,   "d,s,t",    MATCH_FEQ_D, MASK_FEQ_D, match_opcode, 0 },
{"flt.d",      0, INSN_CLASS_D,   "d,S,T",    MATCH_FLT_D, MASK_FLT_D, match_opcode, 0 },
{"flt.d",      0, INSN_CLASS_ZDINX,   "d,s,t",    MATCH_FLT_D, MASK_FLT_D, match_opcode, 0 },
{"fle.d",      0, INSN_CLASS_D,   "d,S,T",    MATCH_FLE_D, MASK_FLE_D, match_opcode, 0 },
{"fle.d",      0, INSN_CLASS_ZDINX,   "d,s,t",    MATCH_FLE_D, MASK_FLE_D, match_opcode, 0 },
{"fgt.d",      0, INSN_CLASS_D,   "d,T,S",    MATCH_FLT_D, MASK_FLT_D, match_opcode, 0 },
{"fgt.d",      0, INSN_CLASS_ZDINX,   "d,t,s",    MATCH_FLT_D, MASK_FLT_D, match_opcode, 0 },
{"fge.d",      0, INSN_CLASS_D,   "d,T,S",    MATCH_FLE_D, MASK_FLE_D, match_opcode, 0 },
{"fge.d",      0, INSN_CLASS_ZDINX,   "d,t,s",    MATCH_FLE_D, MASK_FLE_D, match_opcode, 0 },
{"fmv.x.d",   64, INSN_CLASS_D, "d,S",  MATCH_FMV_X_D, MASK_FMV_X_D, match_opcode, 0 },
{"fmv.d.x",   64, INSN_CLASS_D, "D,s",  MATCH_FMV_D_X, MASK_FMV_D_X, match_opcode, 0 },
{"fcvt.l.d",  64, INSN_CLASS_D, "d,S",  MATCH_FCVT_L_D | MASK_RM, MASK_FCVT_L_D | MASK_RM, match_opcode, 0 },
{"fcvt.l.d",  64, INSN_CLASS_D, "d,S,m",  MATCH_FCVT_L_D, MASK_FCVT_L_D, match_opcode, 0 },
{"fcvt.l.d",  64, INSN_CLASS_ZDINX, "d,s",  MATCH_FCVT_L_D | MASK_RM, MASK_FCVT_L_D | MASK_RM, match_opcode, 0 },
{"fcvt.l.d",  64, INSN_CLASS_ZDINX, "d,s,m",  MATCH_FCVT_L_D, MASK_FCVT_L_D, match_opcode, 0 },
{"fcvt.lu.d", 64, INSN_CLASS_D, "d,S",  MATCH_FCVT_LU_D | MASK_RM, MASK_FCVT_LU_D | MASK_RM, match_opcode, 0 },
{"fcvt.lu.d", 64, INSN_CLASS_D, "d,S,m",  MATCH_FCVT_LU_D, MASK_FCVT_LU_D, match_opcode, 0 },
{"fcvt.lu.d", 64, INSN_CLASS_ZDINX, "d,s",  MATCH_FCVT_LU_D | MASK_RM, MASK_FCVT_LU_D | MASK_RM, match_opcode, 0 },
{"fcvt.lu.d", 64, INSN_CLASS_ZDINX, "d,s,m",  MATCH_FCVT_LU_D, MASK_FCVT_LU_D, match_opcode, 0 },
{"fcvt.d.l",  64, INSN_CLASS_D, "D,s",  MATCH_FCVT_D_L | MASK_RM, MASK_FCVT_D_L | MASK_RM, match_opcode, 0 },
{"fcvt.d.l",  64, INSN_CLASS_D, "D,s,m",  MATCH_FCVT_D_L, MASK_FCVT_D_L, match_opcode, 0 },
{"fcvt.d.l",  64, INSN_CLASS_ZDINX, "d,s",  MATCH_FCVT_D_L | MASK_RM, MASK_FCVT_D_L | MASK_RM, match_opcode, 0 },
{"fcvt.d.l",  64, INSN_CLASS_ZDINX, "d,s,m",  MATCH_FCVT_D_L, MASK_FCVT_D_L, match_opcode, 0 },
{"fcvt.d.lu", 64, INSN_CLASS_D, "D,s",  MATCH_FCVT_D_LU | MASK_RM, MASK_FCVT_D_L | MASK_RM, match_opcode, 0 },
{"fcvt.d.lu", 64, INSN_CLASS_D, "D,s,m",  MATCH_FCVT_D_LU, MASK_FCVT_D_LU, match_opcode, 0 },
{"fcvt.d.lu", 64, INSN_CLASS_ZDINX, "d,s",  MATCH_FCVT_D_LU | MASK_RM, MASK_FCVT_D_L | MASK_RM, match_opcode, 0 },
{"fcvt.d.lu", 64, INSN_CLASS_ZDINX, "d,s,m",  MATCH_FCVT_D_LU, MASK_FCVT_D_LU, match_opcode, 0 },

/* Quad-precision floating-point instruction subset */
{"flq",        0, INSN_CLASS_Q,   "D,o(s)",  MATCH_FLQ, MASK_FLQ, match_opcode, INSN_DREF|INSN_16_BYTE },
{"flq",        0, INSN_CLASS_Q,   "D,A,s",  0, (int) M_FLQ, match_never, INSN_MACRO },
{"fsq",        0, INSN_CLASS_Q,   "T,q(s)",  MATCH_FSQ, MASK_FSQ, match_opcode, INSN_DREF|INSN_16_BYTE },
{"fsq",        0, INSN_CLASS_Q,   "T,A,s",  0, (int) M_FSQ, match_never, INSN_MACRO },
{"fmv.q",      0, INSN_CLASS_Q,   "D,U",  MATCH_FSGNJ_Q, MASK_FSGNJ_Q, match_rs1_eq_rs2, INSN_ALIAS },
{"fneg.q",     0, INSN_CLASS_Q,   "D,U",  MATCH_FSGNJN_Q, MASK_FSGNJN_Q, match_rs1_eq_rs2, INSN_ALIAS },
{"fabs.q",     0, INSN_CLASS_Q,   "D,U",  MATCH_FSGNJX_Q, MASK_FSGNJX_Q, match_rs1_eq_rs2, INSN_ALIAS },
{"fsgnj.q",    0, INSN_CLASS_Q,   "D,S,T",  MATCH_FSGNJ_Q, MASK_FSGNJ_Q, match_opcode, 0 },
{"fsgnjn.q",   0, INSN_CLASS_Q,   "D,S,T",  MATCH_FSGNJN_Q, MASK_FSGNJN_Q, match_opcode, 0 },
{"fsgnjx.q",   0, INSN_CLASS_Q,   "D,S,T",  MATCH_FSGNJX_Q, MASK_FSGNJX_Q, match_opcode, 0 },
{"fadd.q",     0, INSN_CLASS_Q,   "D,S,T",  MATCH_FADD_Q | MASK_RM, MASK_FADD_Q | MASK_RM, match_opcode, 0 },
{"fadd.q",     0, INSN_CLASS_Q,   "D,S,T,m",  MATCH_FADD_Q, MASK_FADD_Q, match_opcode, 0 },
{"fsub.q",     0, INSN_CLASS_Q,   "D,S,T",  MATCH_FSUB_Q | MASK_RM, MASK_FSUB_Q | MASK_RM, match_opcode, 0 },
{"fsub.q",     0, INSN_CLASS_Q,   "D,S,T,m",  MATCH_FSUB_Q, MASK_FSUB_Q, match_opcode, 0 },
{"fmul.q",     0, INSN_CLASS_Q,   "D,S,T",  MATCH_FMUL_Q | MASK_RM, MASK_FMUL_Q | MASK_RM, match_opcode, 0 },
{"fmul.q",     0, INSN_CLASS_Q,   "D,S,T,m",  MATCH_FMUL_Q, MASK_FMUL_Q, match_opcode, 0 },
{"fdiv.q",     0, INSN_CLASS_Q,   "D,S,T",  MATCH_FDIV_Q | MASK_RM, MASK_FDIV_Q | MASK_RM, match_opcode, 0 },
{"fdiv.q",     0, INSN_CLASS_Q,   "D,S,T,m",  MATCH_FDIV_Q, MASK_FDIV_Q, match_opcode, 0 },
{"fsqrt.q",    0, INSN_CLASS_Q,   "D,S",  MATCH_FSQRT_Q | MASK_RM, MASK_FSQRT_Q | MASK_RM, match_opcode, 0 },
{"fsqrt.q",    0, INSN_CLASS_Q,   "D,S,m",  MATCH_FSQRT_Q, MASK_FSQRT_Q, match_opcode, 0 },
{"fmin.q",     0, INSN_CLASS_Q,   "D,S,T",  MATCH_FMIN_Q, MASK_FMIN_Q, match_opcode, 0 },
{"fmax.q",     0, INSN_CLASS_Q,   "D,S,T",  MATCH_FMAX_Q, MASK_FMAX_Q, match_opcode, 0 },
{"fmadd.q",    0, INSN_CLASS_Q,   "D,S,T,R",  MATCH_FMADD_Q | MASK_RM, MASK_FMADD_Q | MASK_RM, match_opcode, 0 },
{"fmadd.q",    0, INSN_CLASS_Q,   "D,S,T,R,m",  MATCH_FMADD_Q, MASK_FMADD_Q, match_opcode, 0 },
{"fnmadd.q",   0, INSN_CLASS_Q,   "D,S,T,R",  MATCH_FNMADD_Q | MASK_RM, MASK_FNMADD_Q | MASK_RM, match_opcode, 0 },
{"fnmadd.q",   0, INSN_CLASS_Q,   "D,S,T,R,m",  MATCH_FNMADD_Q, MASK_FNMADD_Q, match_opcode, 0 },
{"fmsub.q",    0, INSN_CLASS_Q,   "D,S,T,R",  MATCH_FMSUB_Q | MASK_RM, MASK_FMSUB_Q | MASK_RM, match_opcode, 0 },
{"fmsub.q",    0, INSN_CLASS_Q,   "D,S,T,R,m",  MATCH_FMSUB_Q, MASK_FMSUB_Q, match_opcode, 0 },
{"fnmsub.q",   0, INSN_CLASS_Q,   "D,S,T,R",  MATCH_FNMSUB_Q | MASK_RM, MASK_FNMSUB_Q | MASK_RM, match_opcode, 0 },
{"fnmsub.q",   0, INSN_CLASS_Q,   "D,S,T,R,m",  MATCH_FNMSUB_Q, MASK_FNMSUB_Q, match_opcode, 0 },
{"fcvt.w.q",   0, INSN_CLASS_Q,   "d,S",  MATCH_FCVT_W_Q | MASK_RM, MASK_FCVT_W_Q | MASK_RM, match_opcode, 0 },
{"fcvt.w.q",   0, INSN_CLASS_Q,   "d,S,m",  MATCH_FCVT_W_Q, MASK_FCVT_W_Q, match_opcode, 0 },
{"fcvt.wu.q",  0, INSN_CLASS_Q,   "d,S",  MATCH_FCVT_WU_Q | MASK_RM, MASK_FCVT_WU_Q | MASK_RM, match_opcode, 0 },
{"fcvt.wu.q",  0, INSN_CLASS_Q,   "d,S,m",  MATCH_FCVT_WU_Q, MASK_FCVT_WU_Q, match_opcode, 0 },
{"fcvt.q.w",   0, INSN_CLASS_Q,   "D,s",  MATCH_FCVT_Q_W, MASK_FCVT_Q_W | MASK_RM, match_opcode, 0 },
{"fcvt.q.wu",  0, INSN_CLASS_Q,   "D,s",  MATCH_FCVT_Q_WU, MASK_FCVT_Q_WU | MASK_RM, match_opcode, 0 },
{"fcvt.q.s",   0, INSN_CLASS_Q,   "D,S",  MATCH_FCVT_Q_S, MASK_FCVT_Q_S | MASK_RM, match_opcode, 0 },
{"fcvt.q.d",   0, INSN_CLASS_Q,   "D,S",  MATCH_FCVT_Q_D, MASK_FCVT_Q_D | MASK_RM, match_opcode, 0 },
{"fcvt.s.q",   0, INSN_CLASS_Q,   "D,S",  MATCH_FCVT_S_Q | MASK_RM, MASK_FCVT_S_Q | MASK_RM, match_opcode, 0 },
{"fcvt.s.q",   0, INSN_CLASS_Q,   "D,S,m",  MATCH_FCVT_S_Q, MASK_FCVT_S_Q, match_opcode, 0 },
{"fcvt.d.q",   0, INSN_CLASS_Q,   "D,S",  MATCH_FCVT_D_Q | MASK_RM, MASK_FCVT_D_Q | MASK_RM, match_opcode, 0 },
{"fcvt.d.q",   0, INSN_CLASS_Q,   "D,S,m",  MATCH_FCVT_D_Q, MASK_FCVT_D_Q, match_opcode, 0 },
{"fclass.q",   0, INSN_CLASS_Q,   "d,S",  MATCH_FCLASS_Q, MASK_FCLASS_Q, match_opcode, 0 },
{"feq.q",      0, INSN_CLASS_Q,   "d,S,T",    MATCH_FEQ_Q, MASK_FEQ_Q, match_opcode, 0 },
{"flt.q",      0, INSN_CLASS_Q,   "d,S,T",    MATCH_FLT_Q, MASK_FLT_Q, match_opcode, 0 },
{"fle.q",      0, INSN_CLASS_Q,   "d,S,T",    MATCH_FLE_Q, MASK_FLE_Q, match_opcode, 0 },
{"fgt.q",      0, INSN_CLASS_Q,   "d,T,S",    MATCH_FLT_Q, MASK_FLT_Q, match_opcode, 0 },
{"fge.q",      0, INSN_CLASS_Q,   "d,T,S",    MATCH_FLE_Q, MASK_FLE_Q, match_opcode, 0 },
{"fmv.x.q",   64, INSN_CLASS_Q, "d,S",  MATCH_FMV_X_Q, MASK_FMV_X_Q, match_opcode, 0 },
{"fmv.q.x",   64, INSN_CLASS_Q, "D,s",  MATCH_FMV_Q_X, MASK_FMV_Q_X, match_opcode, 0 },
{"fcvt.l.q",  64, INSN_CLASS_Q, "d,S",  MATCH_FCVT_L_Q | MASK_RM, MASK_FCVT_L_Q | MASK_RM, match_opcode, 0 },
{"fcvt.l.q",  64, INSN_CLASS_Q, "d,S,m",  MATCH_FCVT_L_Q, MASK_FCVT_L_Q, match_opcode, 0 },
{"fcvt.lu.q", 64, INSN_CLASS_Q, "d,S",  MATCH_FCVT_LU_Q | MASK_RM, MASK_FCVT_LU_Q | MASK_RM, match_opcode, 0 },
{"fcvt.lu.q", 64, INSN_CLASS_Q, "d,S,m",  MATCH_FCVT_LU_Q, MASK_FCVT_LU_Q, match_opcode, 0 },
{"fcvt.q.l",  64, INSN_CLASS_Q, "D,s",  MATCH_FCVT_Q_L | MASK_RM, MASK_FCVT_Q_L | MASK_RM, match_opcode, 0 },
{"fcvt.q.l",  64, INSN_CLASS_Q, "D,s,m",  MATCH_FCVT_Q_L, MASK_FCVT_Q_L, match_opcode, 0 },
{"fcvt.q.lu", 64, INSN_CLASS_Q, "D,s",  MATCH_FCVT_Q_LU | MASK_RM, MASK_FCVT_Q_L | MASK_RM, match_opcode, 0 },
{"fcvt.q.lu", 64, INSN_CLASS_Q, "D,s,m",  MATCH_FCVT_Q_LU, MASK_FCVT_Q_LU, match_opcode, 0 },

/* Compressed instructions.  */
{"c.unimp",    0, INSN_CLASS_C,   "",  0, 0xffffU,  match_opcode, 0 },
{"c.ebreak",   0, INSN_CLASS_C,   "",  MATCH_C_EBREAK, MASK_C_EBREAK, match_opcode, 0 },
{"c.jr",       0, INSN_CLASS_C,   "d",  MATCH_C_JR, MASK_C_JR, match_rd_nonzero, INSN_BRANCH },
{"c.jalr",     0, INSN_CLASS_C,   "d",  MATCH_C_JALR, MASK_C_JALR, match_rd_nonzero, INSN_JSR },
{"c.j",        0, INSN_CLASS_C,   "Ca",  MATCH_C_J, MASK_C_J, match_opcode, INSN_BRANCH },
{"c.jal",     32, INSN_CLASS_C, "Ca",  MATCH_C_JAL, MASK_C_JAL, match_opcode, INSN_JSR },
{"c.beqz",     0, INSN_CLASS_C,   "Cs,Cp",  MATCH_C_BEQZ, MASK_C_BEQZ, match_opcode, INSN_CONDBRANCH },
{"c.bnez",     0, INSN_CLASS_C,   "Cs,Cp",  MATCH_C_BNEZ, MASK_C_BNEZ, match_opcode, INSN_CONDBRANCH },
{"c.lwsp",     0, INSN_CLASS_C,   "d,Cm(Cc)",  MATCH_C_LWSP, MASK_C_LWSP, match_rd_nonzero, 0 },
{"c.lw",       0, INSN_CLASS_C,   "Ct,Ck(Cs)",  MATCH_C_LW, MASK_C_LW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"c.swsp",     0, INSN_CLASS_C,   "CV,CM(Cc)",  MATCH_C_SWSP, MASK_C_SWSP, match_opcode, INSN_DREF|INSN_4_BYTE },
{"c.sw",       0, INSN_CLASS_C,   "Ct,Ck(Cs)",  MATCH_C_SW, MASK_C_SW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"c.nop",      0, INSN_CLASS_C,   "",  MATCH_C_ADDI, 0xffff, match_opcode, INSN_ALIAS },
{"c.nop",      0, INSN_CLASS_C,   "Cj",  MATCH_C_ADDI, MASK_C_ADDI | MASK_RD, match_opcode, INSN_ALIAS },
{"c.mv",       0, INSN_CLASS_C,   "d,CV",  MATCH_C_MV, MASK_C_MV, match_c_add_with_hint, 0 },
{"c.lui",      0, INSN_CLASS_C,   "d,Cu",  MATCH_C_LUI, MASK_C_LUI, match_c_lui_with_hint, 0 },
{"c.li",       0, INSN_CLASS_C,   "d,Co",  MATCH_C_LI, MASK_C_LI, match_opcode, 0 },
{"c.addi4spn", 0, INSN_CLASS_C,   "Ct,Cc,CK", MATCH_C_ADDI4SPN, MASK_C_ADDI4SPN, match_c_addi4spn, 0 },
{"c.addi16sp", 0, INSN_CLASS_C,   "Cc,CL", MATCH_C_ADDI16SP, MASK_C_ADDI16SP, match_c_addi16sp, 0 },
{"c.addi",     0, INSN_CLASS_C,   "d,Co",  MATCH_C_ADDI, MASK_C_ADDI, match_opcode, 0 },
{"c.add",      0, INSN_CLASS_C,   "d,CV",  MATCH_C_ADD, MASK_C_ADD, match_c_add_with_hint, 0 },
{"c.sub",      0, INSN_CLASS_C,   "Cs,Ct",  MATCH_C_SUB, MASK_C_SUB, match_opcode, 0 },
{"c.and",      0, INSN_CLASS_C,   "Cs,Ct",  MATCH_C_AND, MASK_C_AND, match_opcode, 0 },
{"c.or",       0, INSN_CLASS_C,   "Cs,Ct",  MATCH_C_OR, MASK_C_OR, match_opcode, 0 },
{"c.xor",      0, INSN_CLASS_C,   "Cs,Ct",  MATCH_C_XOR, MASK_C_XOR, match_opcode, 0 },
{"c.slli",     0, INSN_CLASS_C,   "d,C>",  MATCH_C_SLLI, MASK_C_SLLI, match_c_slli, 0 },
{"c.srli",     0, INSN_CLASS_C,   "Cs,C>",  MATCH_C_SRLI, MASK_C_SRLI, match_c_slli, 0 },
{"c.srai",     0, INSN_CLASS_C,   "Cs,C>",  MATCH_C_SRAI, MASK_C_SRAI, match_c_slli, 0 },
{"c.slli64",   0, INSN_CLASS_C,   "d",  MATCH_C_SLLI64, MASK_C_SLLI64, match_c_slli64, 0 },
{"c.srli64",   0, INSN_CLASS_C,   "Cs",  MATCH_C_SRLI64, MASK_C_SRLI64, match_c_slli64, 0 },
{"c.srai64",   0, INSN_CLASS_C,   "Cs",  MATCH_C_SRAI64, MASK_C_SRAI64, match_c_slli64, 0 },
{"c.andi",     0, INSN_CLASS_C,   "Cs,Co",  MATCH_C_ANDI, MASK_C_ANDI, match_opcode, 0 },
{"c.addiw",   64, INSN_CLASS_C, "d,Co",  MATCH_C_ADDIW, MASK_C_ADDIW, match_rd_nonzero, 0 },
{"c.addw",    64, INSN_CLASS_C, "Cs,Ct",  MATCH_C_ADDW, MASK_C_ADDW, match_opcode, 0 },
{"c.subw",    64, INSN_CLASS_C, "Cs,Ct",  MATCH_C_SUBW, MASK_C_SUBW, match_opcode, 0 },
{"c.ldsp",    64, INSN_CLASS_C, "d,Cn(Cc)",  MATCH_C_LDSP, MASK_C_LDSP, match_rd_nonzero, INSN_DREF|INSN_8_BYTE },
{"c.ld",      64, INSN_CLASS_C, "Ct,Cl(Cs)",  MATCH_C_LD, MASK_C_LD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.sdsp",    64, INSN_CLASS_C, "CV,CN(Cc)",  MATCH_C_SDSP, MASK_C_SDSP, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.sd",      64, INSN_CLASS_C, "Ct,Cl(Cs)",  MATCH_C_SD, MASK_C_SD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.fldsp",    0, INSN_CLASS_D_AND_C,   "D,Cn(Cc)",  MATCH_C_FLDSP, MASK_C_FLDSP, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.fld",      0, INSN_CLASS_D_AND_C,   "CD,Cl(Cs)",  MATCH_C_FLD, MASK_C_FLD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.fsdsp",    0, INSN_CLASS_D_AND_C,   "CT,CN(Cc)",  MATCH_C_FSDSP, MASK_C_FSDSP, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.fsd",      0, INSN_CLASS_D_AND_C,   "CD,Cl(Cs)",  MATCH_C_FSD, MASK_C_FSD, match_opcode, INSN_DREF|INSN_8_BYTE },
{"c.flwsp",   32, INSN_CLASS_F_AND_C, "D,Cm(Cc)",  MATCH_C_FLWSP, MASK_C_FLWSP, match_opcode, INSN_DREF|INSN_4_BYTE },
{"c.flw",     32, INSN_CLASS_F_AND_C, "CD,Ck(Cs)",  MATCH_C_FLW, MASK_C_FLW, match_opcode, INSN_DREF|INSN_4_BYTE },
{"c.fswsp",   32, INSN_CLASS_F_AND_C, "CT,CM(Cc)",  MATCH_C_FSWSP, MASK_C_FSWSP, match_opcode, INSN_DREF|INSN_4_BYTE },
{"c.fsw",     32, INSN_CLASS_F_AND_C, "CD,Ck(Cs)",  MATCH_C_FSW, MASK_C_FSW, match_opcode, INSN_DREF|INSN_4_BYTE },

/* Supervisor instructions */
{"csrr",       0, INSN_CLASS_I,   "d,E",  MATCH_CSRRS, MASK_CSRRS | MASK_RS1, match_opcode, INSN_ALIAS },
{"csrwi",      0, INSN_CLASS_I,   "E,Z",  MATCH_CSRRWI, MASK_CSRRWI | MASK_RD, match_opcode, INSN_ALIAS },
{"csrsi",      0, INSN_CLASS_I,   "E,Z",  MATCH_CSRRSI, MASK_CSRRSI | MASK_RD, match_opcode, INSN_ALIAS },
{"csrci",      0, INSN_CLASS_I,   "E,Z",  MATCH_CSRRCI, MASK_CSRRCI | MASK_RD, match_opcode, INSN_ALIAS },
{"csrw",       0, INSN_CLASS_I,   "E,s",  MATCH_CSRRW, MASK_CSRRW | MASK_RD, match_opcode, INSN_ALIAS },
{"csrw",       0, INSN_CLASS_I,   "E,Z",  MATCH_CSRRWI, MASK_CSRRWI | MASK_RD, match_opcode, INSN_ALIAS },
{"csrs",       0, INSN_CLASS_I,   "E,s",  MATCH_CSRRS, MASK_CSRRS | MASK_RD, match_opcode, INSN_ALIAS },
{"csrs",       0, INSN_CLASS_I,   "E,Z",  MATCH_CSRRSI, MASK_CSRRSI | MASK_RD, match_opcode, INSN_ALIAS },
{"csrc",       0, INSN_CLASS_I,   "E,s",  MATCH_CSRRC, MASK_CSRRC | MASK_RD, match_opcode, INSN_ALIAS },
{"csrc",       0, INSN_CLASS_I,   "E,Z",  MATCH_CSRRCI, MASK_CSRRCI | MASK_RD, match_opcode, INSN_ALIAS },
{"csrrwi",     0, INSN_CLASS_I,   "d,E,Z",  MATCH_CSRRWI, MASK_CSRRWI, match_opcode, 0 },
{"csrrsi",     0, INSN_CLASS_I,   "d,E,Z",  MATCH_CSRRSI, MASK_CSRRSI, match_opcode, 0 },
{"csrrci",     0, INSN_CLASS_I,   "d,E,Z",  MATCH_CSRRCI, MASK_CSRRCI, match_opcode, 0 },
{"csrrw",      0, INSN_CLASS_I,   "d,E,s",  MATCH_CSRRW, MASK_CSRRW, match_opcode, 0 },
{"csrrw",      0, INSN_CLASS_I,   "d,E,Z",  MATCH_CSRRWI, MASK_CSRRWI, match_opcode, INSN_ALIAS },
{"csrrs",      0, INSN_CLASS_I,   "d,E,s",  MATCH_CSRRS, MASK_CSRRS, match_opcode, 0 },
{"csrrs",      0, INSN_CLASS_I,   "d,E,Z",  MATCH_CSRRSI, MASK_CSRRSI, match_opcode, INSN_ALIAS },
{"csrrc",      0, INSN_CLASS_I,   "d,E,s",  MATCH_CSRRC, MASK_CSRRC, match_opcode, 0 },
{"csrrc",      0, INSN_CLASS_I,   "d,E,Z",  MATCH_CSRRCI, MASK_CSRRCI, match_opcode, INSN_ALIAS },
{"uret",       0, INSN_CLASS_I,   "",     MATCH_URET, MASK_URET, match_opcode, 0 },
{"sret",       0, INSN_CLASS_I,   "",     MATCH_SRET, MASK_SRET, match_opcode, 0 },
{"hret",       0, INSN_CLASS_I,   "",     MATCH_HRET, MASK_HRET, match_opcode, 0 },
{"mret",       0, INSN_CLASS_I,   "",     MATCH_MRET, MASK_MRET, match_opcode, 0 },
{"dret",       0, INSN_CLASS_I,   "",     MATCH_DRET, MASK_DRET, match_opcode, 0 },
{"sfence.vm",  0, INSN_CLASS_I,   "",     MATCH_SFENCE_VM, MASK_SFENCE_VM | MASK_RS1, match_opcode, 0 },
{"sfence.vm",  0, INSN_CLASS_I,   "s",    MATCH_SFENCE_VM, MASK_SFENCE_VM, match_opcode, 0 },
{"sfence.vma", 0, INSN_CLASS_I,   "",     MATCH_SFENCE_VMA, MASK_SFENCE_VMA | MASK_RS1 | MASK_RS2, match_opcode, INSN_ALIAS },
{"sfence.vma", 0, INSN_CLASS_I,   "s",    MATCH_SFENCE_VMA, MASK_SFENCE_VMA | MASK_RS2, match_opcode, INSN_ALIAS },
{"sfence.vma", 0, INSN_CLASS_I,   "s,t",  MATCH_SFENCE_VMA, MASK_SFENCE_VMA, match_opcode, 0 },
{"wfi",        0, INSN_CLASS_I,   "",     MATCH_WFI, MASK_WFI, match_opcode, 0 },


/* PULP specific opcodes */

{"scallimm",    0, INSN_CLASS_I, "b3", MATCH_SCALL, MASK_SCALL_IMM, match_opcode,   0 },

/*
  | extensions/arch       | pulpv0 | pulpv1 | pulpv2 | pulpv3 | gap8 | pulpnn | gap9 | corev |
  |-----------------------+--------+--------+--------+--------+------+--------+------+-------|
  | PULP_POSTMOD          | x (c)  | x (C)  | x      | x      | x    | x      | x    | x     |
  | PULP_INDREGREG        | x      | x      | x      | x      | x    | x      | x    | x     |
  | PULP_ABS              | x (c)  | x (c)  | x      | x      | x    | x      | x    | x     |
  | PULP_SLET             | x      | x      | x      | x      | x    | x      | x    | x     |
  | PULP_MINMAX           | x (c)  | x (c)  | x      | x      | x    | x      | x    | x     |
  | PULP_BITOP_SMALL      | x      | x      |        |        |      |        |      |       |
  | PULP_BITOP            |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_CLIP             |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_HWLOOP           | x (d)  | x      | x      | x      | x    | x      | x    | x     |
  | PULP_MAC_ALT          | x      | x      |        |        |      |        |      |       |
  | PULP_MAC_SI           |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_MACRN_HI         |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_MULRN_HI         |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_PARTMAC          |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_ADDSUBRN         |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_VECT             |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_VECT_SHUFFLEPACK |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_VECT_COMPLEX     |        |        |        | x      |      | x      |      | x     |
  | PULP_VECT_GAP8        |        |        |        |        | x    |        |      |       |
  | PULP_VECT_GAP9        |        |        |        |        |      |        | x    |       |
  | PULP_BR               |        |        | x      | x      | x    | x      | x    | x     |
  | PULP_ELW              | x      | x      | x      | x      | x    | x      | x    | x     |
  | PULP_NN               |        |        |        |        |      | x      |      |       |
  | PULP_BITREV           |        |        |        | x      |      |        | x    |       |
  | PULP_FINX_GAP9        |        |        |        |        |      |        | x    |       |
  | PULP_HALFFLOAT_GAP9   |        |        |        |        |      |        | x    |       |

  c = compatibility mode
  d = disabled in compiler due to bugs
*/

/* for PULP_SLIM just use m-extension plus elw extension */
/* PULP postmod compatibility for pulpv0 and pulpv1 */
/* post-increment and register-register loads */
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s)",       MATCH_LB,                               MASK_LB,        match_opcode,   0},
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s!)",      MATCH_LBPOST,                           MASK_LPOST,     match_opcode,   0},
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s)",       MATCH_LBRR,                             MASK_LRR,       match_opcode,   0},
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s!)",      MATCH_LBRRPOST,                         MASK_LRRPOST,   match_opcode,   0},

{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s)",       MATCH_LBU,                              MASK_LBU,       match_opcode,   0},
{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s!)",      MATCH_LBUPOST,                          MASK_LPOST,     match_opcode,   0},
{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s)",       MATCH_LBURR,                            MASK_LRR,       match_opcode,   0},
{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s!)",      MATCH_LBURRPOST,                        MASK_LRRPOST,   match_opcode,   0},

{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s)",       MATCH_LH,                               MASK_LH,        match_opcode,   0},
{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s!)",      MATCH_LHPOST,                           MASK_LPOST,     match_opcode,   0},
{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s)",       MATCH_LHRR,                             MASK_LRR,       match_opcode,   0},
{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s!)",      MATCH_LHRRPOST,                         MASK_LRRPOST,   match_opcode,   0},

{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s)",       MATCH_LHU,                              MASK_LHU,       match_opcode,   0},
{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s!)",      MATCH_LHUPOST,                          MASK_LPOST,     match_opcode,   0},
{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s)",       MATCH_LHURR,                            MASK_LRR,       match_opcode,   0},
{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s!)",      MATCH_LHURRPOST,                        MASK_LRRPOST,   match_opcode,   0},

{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s)",       MATCH_LW,                               MASK_LW,        match_opcode,   0},
{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,o(s!)",      MATCH_LWPOST,                           MASK_LPOST,     match_opcode,   0},
{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s)",       MATCH_LWRR,                             MASK_LRR,       match_opcode,   0},
{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "d,t(s!)",      MATCH_LWRRPOST,                         MASK_LRRPOST,   match_opcode,   0},

/* post-increment and reg-reg stores */
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,q(s)",       MATCH_SB,                               MASK_SB,        match_opcode,   0},
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,q(s!)",      MATCH_SBPOST,                           MASK_SPOST,     match_opcode,   0},
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,y(s)",       MATCH_SBRR,                             MASK_SRR,       match_opcode,   0},
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,y(s!)",      MATCH_SBRRPOST,                         MASK_SRRPOST,   match_opcode,   0},

{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,q(s)",       MATCH_SH,                               MASK_SH,        match_opcode,   0},
{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,q(s!)",      MATCH_SHPOST,                           MASK_SPOST,     match_opcode,   0},
{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,y(s)",       MATCH_SHRR,                             MASK_SRR,       match_opcode,   0},
{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,y(s!)",      MATCH_SHRRPOST,                         MASK_SRRPOST,   match_opcode,   0},

{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,q(s)",       MATCH_SW,                               MASK_SW,        match_opcode,   0},
{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,q(s!)",      MATCH_SWPOST,                           MASK_SPOST,     match_opcode,   0},
{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,y(s)",       MATCH_SWRR,                             MASK_SRR,       match_opcode,   0},
{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD_COMPAT, "t,y(s!)",      MATCH_SWRRPOST,                         MASK_SRRPOST,   match_opcode,   0},

/* PULP minmax compatibility for pulpv0 and pulpv1 */
{"p.avg",               0, INSN_CLASS_XPULP_MINMAX_COMPAT, "d,s,t",        MATCH_AVG,                              MASK_PALU,      match_opcode,   0},
{"p.avgu",              0, INSN_CLASS_XPULP_MINMAX_COMPAT, "d,s,t",        MATCH_AVGU,                             MASK_PALU,      match_opcode,   0},
{"p.min",               0, INSN_CLASS_XPULP_MINMAX_COMPAT, "d,s,t",        MATCH_MIN,                              MASK_PALU,      match_opcode,   0},
{"p.minu",              0, INSN_CLASS_XPULP_MINMAX_COMPAT, "d,s,t",        MATCH_MINU,                             MASK_PALU,      match_opcode,   0},
{"p.max",               0, INSN_CLASS_XPULP_MINMAX_COMPAT, "d,s,t",        MATCH_MAX,                              MASK_PALU,      match_opcode,   0},
{"p.maxu",              0, INSN_CLASS_XPULP_MINMAX_COMPAT, "d,s,t",        MATCH_MAXU,                             MASK_PALU,      match_opcode,   0},

/* PULP mac alt for pulpv0 and pulpv1. Some guesswork was involved which needs to be verified */
/* 32x32 into 32 MAC operation */
{"p.mac",               0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MAC,                              MASK_MAC,       match_opcode,   0},

/* 16x16 into 32 MAC operations */
{"p.mac.zl.zl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZLZL,                          MASK_MAC,       match_opcode,   0},
{"p.macu",              0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZLZL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.zl.zh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZLZH,                          MASK_MAC,       match_opcode,   0},
{"p.mac.zh.zl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZHZL,                          MASK_MAC,       match_opcode,   0},
{"p.machlu",            0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZHZL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.zh.zh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZHZH,                          MASK_MAC,       match_opcode,   0},
{"p.machhu",            0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZHZH,                          MASK_MAC,       match_opcode,   0},

{"p.mac.zl.sl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZLSL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.zl.sh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZLSH,                          MASK_MAC,       match_opcode,   0},
{"p.mac.zh.sl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZHSL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.zh.sh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACZHSH,                          MASK_MAC,       match_opcode,   0},

{"p.mac.sl.zl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSLZL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.sl.zh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSLZH,                          MASK_MAC,       match_opcode,   0},
{"p.mac.sh.zl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHZL,                          MASK_MAC,       match_opcode,   0},
{"p.machlsu",           0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHZL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.sh.zh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHZH,                          MASK_MAC,       match_opcode,   0},

{"p.mac.sl.sl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSLSL,                          MASK_MAC,       match_opcode,   0},
{"p.macs",              0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSLSL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.sl.sh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSLSH,                          MASK_MAC,       match_opcode,   0},
{"p.mac.sh.sl",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHSL,                          MASK_MAC,       match_opcode,   0},
{"p.machls",            0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHSL,                          MASK_MAC,       match_opcode,   0},
{"p.mac.sh.sh",         0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHSH,                          MASK_MAC,       match_opcode,   0},
{"p.machhs",            0, INSN_CLASS_XPULP_MAC_ALT, "d,s,t,y",      MATCH_MACSHSH,                          MASK_MAC,       match_opcode,   0},

/* PULP abs compatibility for pulpv0 and pulpv1 */
{"p.abs",               0, INSN_CLASS_XPULP_ABS_COMPAT, "d,s",          MATCH_ABS,                              MASK_PALUS,     match_opcode,   0},

/* PULP postmod */
/* post-increment and register-register loads */
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD, "d,o(s)",       MATCH_LB,                               MASK_LB,        match_opcode,   0},
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD, "d,o(s!)",      MATCH_LBPOST,                           MASK_LPOST,     match_opcode,   0},
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD, "d,t(s)",       MATCH_LBRR,                             MASK_LRR,       match_opcode,   0},
{"p.lb",                0, INSN_CLASS_XPULP_POSTMOD, "d,t(s!)",      MATCH_LBRRPOST,                         MASK_LRRPOST,   match_opcode,   0},

{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD, "d,o(s)",       MATCH_LBU,                              MASK_LBU,       match_opcode,   0},
{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD, "d,o(s!)",      MATCH_LBUPOST,                          MASK_LPOST,     match_opcode,   0},
{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD, "d,t(s)",       MATCH_LBURR,                            MASK_LRR,       match_opcode,   0},
{"p.lbu",               0, INSN_CLASS_XPULP_POSTMOD, "d,t(s!)",      MATCH_LBURRPOST,                        MASK_LRRPOST,   match_opcode,   0},

{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD, "d,o(s)",       MATCH_LH,                               MASK_LH,        match_opcode,   0},
{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD, "d,o(s!)",      MATCH_LHPOST,                           MASK_LPOST,     match_opcode,   0},
{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD, "d,t(s)",       MATCH_LHRR,                             MASK_LRR,       match_opcode,   0},
{"p.lh",                0, INSN_CLASS_XPULP_POSTMOD, "d,t(s!)",      MATCH_LHRRPOST,                         MASK_LRRPOST,   match_opcode,   0},

{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD, "d,o(s)",       MATCH_LHU,                              MASK_LHU,       match_opcode,   0},
{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD, "d,o(s!)",      MATCH_LHUPOST,                          MASK_LPOST,     match_opcode,   0},
{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD, "d,t(s)",       MATCH_LHURR,                            MASK_LRR,       match_opcode,   0},
{"p.lhu",               0, INSN_CLASS_XPULP_POSTMOD, "d,t(s!)",      MATCH_LHURRPOST,                        MASK_LRRPOST,   match_opcode,   0},

{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD, "d,o(s)",       MATCH_LW,                               MASK_LW,        match_opcode,   0},
{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD, "d,o(s!)",      MATCH_LWPOST,                           MASK_LPOST,     match_opcode,   0},
{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD, "d,t(s)",       MATCH_LWRR,                             MASK_LRR,       match_opcode,   0},
{"p.lw",                0, INSN_CLASS_XPULP_POSTMOD, "d,t(s!)",      MATCH_LWRRPOST,                         MASK_LRRPOST,   match_opcode,   0},

/* post-increment and reg-reg stores */
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD, "t,q(s)",       MATCH_SB,                               MASK_SB,        match_opcode,   0},
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD, "t,q(s!)",      MATCH_SBPOST,                           MASK_SPOST,     match_opcode,   0},
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD, "t,d(s)",       MATCH_SBRR,                             MASK_PALU,      match_opcode,   0},
{"p.sb",                0, INSN_CLASS_XPULP_POSTMOD, "t,d(s!)",      MATCH_SBRRPOST,                         MASK_PALU,      match_opcode,   0},

{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD, "t,q(s)",       MATCH_SH,                               MASK_SH,        match_opcode,   0},
{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD, "t,q(s!)",      MATCH_SHPOST,                           MASK_SPOST,     match_opcode,   0},
{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD, "t,d(s)",       MATCH_SHRR,                             MASK_PALU,      match_opcode,   0},
{"p.sh",                0, INSN_CLASS_XPULP_POSTMOD, "t,d(s!)",      MATCH_SHRRPOST,                         MASK_PALU,      match_opcode,   0},

{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD, "t,q(s)",       MATCH_SW,                               MASK_SW,        match_opcode,   0},
{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD, "t,q(s!)",      MATCH_SWPOST,                           MASK_SPOST,     match_opcode,   0},
{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD, "t,d(s)",       MATCH_SWRR,                             MASK_PALU,      match_opcode,   0},
{"p.sw",                0, INSN_CLASS_XPULP_POSTMOD, "t,d(s!)",      MATCH_SWRRPOST,                         MASK_PALU,      match_opcode,   0},

/* PULP abs */
{"p.abs",               0, INSN_CLASS_XPULP_ABS, "d,s",          MATCH_AVG,/* remapped (!)*/                 MASK_PALUS,     match_opcode,   0},

/* PULP slet */
{"p.slet",              0, INSN_CLASS_XPULP_SLET, "d,s,t",        MATCH_SLET,                             MASK_PALU,      match_opcode,   0},
{"p.sletu",             0, INSN_CLASS_XPULP_SLET, "d,s,t",        MATCH_SLETU,                            MASK_PALU,      match_opcode,   0},

/* PULP min/max */
{"p.min",               0, INSN_CLASS_XPULP_MINMAX, "d,s,t",        MATCH_MIN,                              MASK_PALU,      match_opcode,   0},
{"p.minu",              0, INSN_CLASS_XPULP_MINMAX, "d,s,t",        MATCH_MINU,                             MASK_PALU,      match_opcode,   0},
{"p.max",               0, INSN_CLASS_XPULP_MINMAX, "d,s,t",        MATCH_MAX,                              MASK_PALU,      match_opcode,   0},
{"p.maxu",              0, INSN_CLASS_XPULP_MINMAX, "d,s,t",        MATCH_MAXU,                             MASK_PALU,      match_opcode,   0},

/* PULP bitop small */
/* bit fiddling */
{"p.ff1",               0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_FF1,                              MASK_PALUS,     match_opcode,   0},
{"p.fl1",               0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_FL1,                              MASK_PALUS,     match_opcode,   0},
{"p.clb",               0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_CLB,                              MASK_PALUS,     match_opcode,   0},
{"p.cnt",               0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_CNT,                              MASK_PALUS,     match_opcode,   0},
{"p.ror",               0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s,t",        MATCH_ROR,                              MASK_PALU,      match_opcode,   0},
/* sign extensions */
{"p.exths",             0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_EXTHS,                            MASK_PALUS,     match_opcode,   0},
{"p.exthz",             0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_EXTHZ,                            MASK_PALUS,     match_opcode,   0},
{"p.extbs",             0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_EXTBS,                            MASK_PALUS,     match_opcode,   0},
{"p.extbz",             0, INSN_CLASS_XPULP_BITOP_SMALL, "d,s",          MATCH_EXTBZ,                            MASK_PALUS,     match_opcode,   0},

/* PULP bitop */
/* bit fiddling */
{"p.ff1",               0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_FF1,                              MASK_PALUS,     match_opcode,   0},
{"p.fl1",               0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_FL1,                              MASK_PALUS,     match_opcode,   0},
{"p.clb",               0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_CLB,                              MASK_PALUS,     match_opcode,   0},
{"p.cnt",               0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_CNT,                              MASK_PALUS,     match_opcode,   0},
{"p.ror",               0, INSN_CLASS_XPULP_BITOP, "d,s,t",        MATCH_ROR,                              MASK_PALU,      match_opcode,   0},
/* sign extensions */
{"p.exths",             0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_EXTHS,                            MASK_PALUS,     match_opcode,   0},
{"p.exthz",             0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_EXTHZ,                            MASK_PALUS,     match_opcode,   0},
{"p.extbs",             0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_EXTBS,                            MASK_PALUS,     match_opcode,   0},
{"p.extbz",             0, INSN_CLASS_XPULP_BITOP, "d,s",          MATCH_EXTBZ,                            MASK_PALUS,     match_opcode,   0},
/* set/clear etc. */
{"p.extract",           0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_EXTRACT,                          MASK_PALU2,     match_opcode,   0},
{"p.extracti",          0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_EXTRACT,                          MASK_PALU2,     match_opcode,   0},
{"p.extractr",          0, INSN_CLASS_XPULP_BITOP, "d,s,t",        MATCH_EXTRACTR,                         MASK_PALU,      match_opcode,   0},
{"p.extractu",          0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_EXTRACTU,                         MASK_PALU2,     match_opcode,   0},
{"p.extractui",         0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_EXTRACTU,                         MASK_PALU2,     match_opcode,   0},
{"p.extractur",         0, INSN_CLASS_XPULP_BITOP, "d,s,t",        MATCH_EXTRACTUR,                        MASK_PALU,      match_opcode,   0},
{"p.insert",            0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_INSERT,                           MASK_PALU2,     match_opcode,   0},
{"p.inserti",           0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_INSERT,                           MASK_PALU2,     match_opcode,   0},
{"p.insertr",           0, INSN_CLASS_XPULP_BITOP, "d,s,t",        MATCH_INSERTR,                          MASK_PALU,      match_opcode,   0},
{"p.bset",              0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_BSET,                             MASK_PALU2,     match_opcode,   0},
{"p.bseti",             0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_BSET,                             MASK_PALU2,     match_opcode,   0},
{"p.bsetr",             0, INSN_CLASS_XPULP_BITOP, "d,s,t",        MATCH_BSETR,                            MASK_PALU,      match_opcode,   0},
{"p.bclr",              0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_BCLR,                             MASK_PALU2,     match_opcode,   0},
{"p.bclri",             0, INSN_CLASS_XPULP_BITOP, "d,s,b5,bi",    MATCH_BCLR,                             MASK_PALU2,     match_opcode,   0},
{"p.bclrr",             0, INSN_CLASS_XPULP_BITOP, "d,s,t",        MATCH_BCLRR,                            MASK_PALU,      match_opcode,   0},

/* PULP clip */
{"p.clip",              0, INSN_CLASS_XPULP_CLIP, "d,s,bi",       MATCH_CLIP,                             MASK_PALU1,     match_opcode,   0},
{"p.clipi",             0, INSN_CLASS_XPULP_CLIP, "d,s,bi",       MATCH_CLIP,                             MASK_PALU1,     match_opcode,   0},
{"p.clipr",             0, INSN_CLASS_XPULP_CLIP, "d,s,t",        MATCH_CLIPR,                            MASK_PALU,      match_opcode,   0},
{"p.clipu",             0, INSN_CLASS_XPULP_CLIP, "d,s,bi",       MATCH_CLIPU,                            MASK_PALU1,     match_opcode,   0},
{"p.clipui",            0, INSN_CLASS_XPULP_CLIP, "d,s,bi",       MATCH_CLIPU,                            MASK_PALU1,     match_opcode,   0},
{"p.clipur",            0, INSN_CLASS_XPULP_CLIP, "d,s,t",        MATCH_CLIPUR,                           MASK_PALU,      match_opcode,   0},

/* PULP hwloop */
{"lp.starti",           0, INSN_CLASS_XPULP_HWLOOP, "di,b1",        MATCH_HWLP_STARTI,                      MASK_HWLP_STARTI,match_opcode,  0},
{"lp.endi",             0, INSN_CLASS_XPULP_HWLOOP, "di,b1",        MATCH_HWLP_ENDI,                        MASK_HWLP_ENDI,  match_opcode,  0},
{"lp.count",            0, INSN_CLASS_XPULP_HWLOOP, "di,s",         MATCH_HWLP_COUNT,                       MASK_HWLP_COUNT, match_opcode,  0},
{"lp.counti",           0, INSN_CLASS_XPULP_HWLOOP, "di,ji",        MATCH_HWLP_COUNTI,                      MASK_HWLP_COUNTI,match_opcode,  0},
{"lp.setup",            0, INSN_CLASS_XPULP_HWLOOP, "di,s,b1",      MATCH_HWLP_SETUP,                       MASK_HWLP_SETUP, match_opcode,  0},
{"lp.setupi",           0, INSN_CLASS_XPULP_HWLOOP, "di,ji,b2",     MATCH_HWLP_SETUPI,                      MASK_HWLP_SETUPI,match_opcode,  0},


/* PULP addsubrn */
/* add/sub with norm and rounding */
{"p.addn",              0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDN,                             MASK_MACMULNR,  match_opcode,   0},
{"p.addni",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDN,                             MASK_MACMULNR,  match_opcode,   0},
{"p.addnr",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_ADDNR,                            MASK_PALU,      match_opcode,   0},
{"p.addun",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDUN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.adduni",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDUN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.addunr",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_ADDUNR,                           MASK_PALU,      match_opcode,   0},
{"p.addrn",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDRN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.addrni",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDRN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.addrnr",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_ADDRNR,                           MASK_PALU,      match_opcode,   0},
{"p.addurn",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDURN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.addurni",           0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_ADDURN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.addurnr",           0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_ADDURNR,                          MASK_PALU,      match_opcode,   0},

{"p.subn",              0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBN,                             MASK_MACMULNR,  match_opcode,   0},
{"p.subni",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBN,                             MASK_MACMULNR,  match_opcode,   0},
{"p.subnr",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_SUBNR,                            MASK_PALU,      match_opcode,   0},
{"p.subun",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBUN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.subuni",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBUN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.subunr",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_SUBUNR,                           MASK_PALU,      match_opcode,   0},
{"p.subrn",             0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBRN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.subrni",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBRN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.subrnr",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_SUBRNR,                           MASK_PALU,      match_opcode,   0},
{"p.suburn",            0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBURN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.suburni",           0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t,b5",     MATCH_SUBURN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.suburnr",           0, INSN_CLASS_XPULP_ADDSUBRN, "d,s,t",        MATCH_SUBURNR,                          MASK_PALU,      match_opcode,   0},

/* PULP mac */

/* partmac 6x16 into 32 */
{"p.macs",              0, INSN_CLASS_XPULP_PARTMAC, "d,s,t",        MATCH_MACS,                             MASK_MACMUL,    match_opcode,   0},
{"p.machhs",            0, INSN_CLASS_XPULP_PARTMAC, "d,s,t",        MATCH_MACHHS,                           MASK_MACMUL,    match_opcode,   0},
{"p.macu",              0, INSN_CLASS_XPULP_PARTMAC, "d,s,t",        MATCH_MACU,                             MASK_MACMUL,    match_opcode,   0},
{"p.machhu",            0, INSN_CLASS_XPULP_PARTMAC, "d,s,t",        MATCH_MACHHU,                           MASK_MACMUL,    match_opcode,   0},

/* Mac/Msu 32x32 into 32 */
{"p.mac",               0, INSN_CLASS_XPULP_MAC_SI, "d,s,t",        MATCH_MAC32,                            MASK_MACMSU32,  match_opcode,   0},
{"p.msu",               0, INSN_CLASS_XPULP_MAC_SI, "d,s,t",        MATCH_MSU32,                            MASK_MACMSU32,  match_opcode,   0},

/* Mac with norm/round 16x16 into 32 */
{"p.macsn",             0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACSN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.machhsn",           0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACHHSN,                          MASK_MACMULNR,  match_opcode,   0},
{"p.macsrn",            0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACSRN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.machhsrn",          0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACHHSRN,                         MASK_MACMULNR,  match_opcode,   0},

{"p.macun",             0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACUN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.machhun",           0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACHHUN,                          MASK_MACMULNR,  match_opcode,   0},
{"p.macurn",            0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACURN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.machhurn",          0, INSN_CLASS_XPULP_MACRN_HI, "d,s,t,b5",     MATCH_MACHHURN,                         MASK_MACMULNR,  match_opcode,   0},

/* Mul with norm/round 16x16 into 32 */
/* these need to get before the one with the postfix 'n' */
{"p.muls",              0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t",        MATCH_MULS,                             MASK_MACMUL,    match_opcode,   0},
{"p.mulhhs",            0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t",        MATCH_MULHHS,                           MASK_MACMUL,    match_opcode,   0},
{"p.mulu",              0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t",        MATCH_MULU,                             MASK_MACMUL,    match_opcode,   0},
{"p.mulhhu",            0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t",        MATCH_MULHHU,                           MASK_MACMUL,    match_opcode,   0},

{"p.mulsn",             0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULSN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.mulhhsn",           0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULHHSN,                          MASK_MACMULNR,  match_opcode,   0},
{"p.mulsrn",            0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULSRN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.mulhhsrn",          0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULHHSRN,                         MASK_MACMULNR,  match_opcode,   0},

{"p.mulun",             0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULUN,                            MASK_MACMULNR,  match_opcode,   0},
{"p.mulhhun",           0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULHHUN,                          MASK_MACMULNR,  match_opcode,   0},
{"p.mulurn",            0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULURN,                           MASK_MACMULNR,  match_opcode,   0},
{"p.mulhhurn",          0, INSN_CLASS_XPULP_MULRN_HI, "d,s,t,b5",     MATCH_MULHHURN,                         MASK_MACMULNR,  match_opcode,   0},


/* just use standard RISC-V mul extension */
// /* 32x32 into 32 multiplication */
// {"p.mul",               0, INSN_CLASS_XPULP_V2, "d,s,t",        MATCH_MUL,                              MASK_MUL,       match_opcode,   0},
// /* 32x32 into 64 support */
// {"p.mulh",              0, INSN_CLASS_XPULP_V2, "d,s,t",        MATCH_MULH,                             MASK_MULH,      match_opcode,  0},
// {"p.mulhu",             0, INSN_CLASS_XPULP_V2, "d,s,t",        MATCH_MULHU,                            MASK_MULHU,     match_opcode,  0},
// {"p.mulhsu",            0, INSN_CLASS_XPULP_V2, "d,s,t",        MATCH_MULHSU,                           MASK_MULHSU,    match_opcode,  0},
// /* 32 bit div and rem */
// {"p.div",               0, INSN_CLASS_XPULP_V2,   "d,s,t",      MATCH_DIV,                              MASK_DIV,       match_opcode,  0},
// {"p.divu",              0, INSN_CLASS_XPULP_V2,   "d,s,t",      MATCH_DIVU,                             MASK_DIVU,      match_opcode,  0},
// {"p.rem",               0, INSN_CLASS_XPULP_V2,   "d,s,t",      MATCH_REM,                              MASK_REM,       match_opcode,  0},
// {"p.remu",              0, INSN_CLASS_XPULP_V2,   "d,s,t",      MATCH_REMU,                             MASK_REMU,      match_opcode,  0},


/* PULP vect */
/* Vector/SIMD Operations */
{"pv.add.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.add.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.add.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_ADD|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.add.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.add.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.add.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_ADD|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.sub.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_SUB|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.sub.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_SUB|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.avg.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_AVG|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.avg.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_AVG|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.avgu.h",           0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_H_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.sc.h",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_H_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.sci.h",       0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_AVGU|MATCH_V_OP_H_VI,        MASK_V_OP1,     match_opcode,   0},
{"pv.avgu.b",           0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_B_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.sc.b",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_B_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.sci.b",       0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_AVGU|MATCH_V_OP_B_VI,        MASK_V_OP1,     match_opcode,   0},

{"pv.min.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.min.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.min.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_MIN|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.min.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.min.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.min.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_MIN|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.minu.h",           0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_H_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.sc.h",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_H_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.sci.h",       0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_MINU|MATCH_V_OP_H_VI,        MASK_V_OP1,     match_opcode,   0},
{"pv.minu.b",           0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_B_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.sc.b",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_B_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.sci.b",       0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_MINU|MATCH_V_OP_B_VI,        MASK_V_OP1,     match_opcode,   0},

{"pv.max.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.max.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.max.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_MAX|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.max.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.max.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.max.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_MAX|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.maxu.h",           0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_H_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.sc.h",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_H_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.sci.h",       0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_MAXU|MATCH_V_OP_H_VI,        MASK_V_OP1,     match_opcode,   0},
{"pv.maxu.b",           0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_B_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.sc.b",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_B_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.sci.b",       0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_MAXU|MATCH_V_OP_B_VI,        MASK_V_OP1,     match_opcode,   0},

{"pv.srl.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bU",       MATCH_V_OP_SRL|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.srl.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bU",       MATCH_V_OP_SRL|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.sra.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bU",       MATCH_V_OP_SRA|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.sra.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bU",       MATCH_V_OP_SRA|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.sll.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bU",       MATCH_V_OP_SLL|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.sll.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bU",       MATCH_V_OP_SLL|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.or.h",             0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_H_VV,          MASK_V_OP,      match_opcode,   0},
{"pv.or.sc.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_H_VR,          MASK_V_OP,      match_opcode,   0},
{"pv.or.sci.h",         0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_OR|MATCH_V_OP_H_VI,          MASK_V_OP1,     match_opcode,   0},
{"pv.or.b",             0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_B_VV,          MASK_V_OP,      match_opcode,   0},
{"pv.or.sc.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_B_VR,          MASK_V_OP,      match_opcode,   0},
{"pv.or.sci.b",         0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_OR|MATCH_V_OP_B_VI,          MASK_V_OP1,     match_opcode,   0},

{"pv.xor.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_XOR|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.xor.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_XOR|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.and.h",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_H_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.and.sc.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_H_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.and.sci.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_AND|MATCH_V_OP_H_VI,         MASK_V_OP1,     match_opcode,   0},
{"pv.and.b",            0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_B_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.and.sc.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_B_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.and.sci.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_AND|MATCH_V_OP_B_VI,         MASK_V_OP1,     match_opcode,   0},

{"pv.abs.h",            0, INSN_CLASS_XPULP_VECT, "d,s",          MATCH_V_OP_ABS|MATCH_V_OP_H_VV,         MASK_V_OP2,     match_opcode,   0},
{"pv.abs.b",            0, INSN_CLASS_XPULP_VECT, "d,s",          MATCH_V_OP_ABS|MATCH_V_OP_B_VV,         MASK_V_OP2,     match_opcode,   0},

{"pv.extract.h",        0, INSN_CLASS_XPULP_VECT, "d,s,bf",       MATCH_V_OP_EXTRACT|MATCH_V_OP_H_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.extract.b",        0, INSN_CLASS_XPULP_VECT, "d,s,bF",       MATCH_V_OP_EXTRACT|MATCH_V_OP_B_VI,     MASK_V_OP1,     match_opcode,   0},

{"pv.extractu.h",       0, INSN_CLASS_XPULP_VECT, "d,s,bf",       MATCH_V_OP_DOTSP|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.extractu.b",       0, INSN_CLASS_XPULP_VECT, "d,s,bF",       MATCH_V_OP_DOTSP|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.insert.h",         0, INSN_CLASS_XPULP_VECT, "d,s,bf",       MATCH_V_OP_SDOTUP|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.insert.b",         0, INSN_CLASS_XPULP_VECT, "d,s,bF",       MATCH_V_OP_SDOTUP|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.dotsp.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_DOTUP|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.dotsp.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_DOTUP|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.dotup.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_H_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_H_VR,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_EXTRACTU|MATCH_V_OP_H_VI,    MASK_V_OP1,     match_opcode,   0},
{"pv.dotup.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_B_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_B_VR,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_EXTRACTU|MATCH_V_OP_B_VI,    MASK_V_OP1,     match_opcode,   0},

{"pv.dotusp.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_INSERT|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.dotusp.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_INSERT|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.sdotsp.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_H_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_H_VR,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_SDOTUSP|MATCH_V_OP_H_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.sdotsp.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_B_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_B_VR,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_SDOTUSP|MATCH_V_OP_B_VI,     MASK_V_OP1,     match_opcode,   0},

{"pv.sdotup.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_DOTUSP|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.sdotup.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_DOTUSP|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.sdotusp.h",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.sc.h",     0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.sci.h",    0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_SDOTSP|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.sdotusp.b",        0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.sc.b",     0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.sci.b",    0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_SDOTSP|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

/* vector/SIMD compares */
{"pv.cmpeq.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPEQ|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpeq.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPEQ|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpeq.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPEQ|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.cmpeq.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPEQ|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpeq.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPEQ|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpeq.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPEQ|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.cmpne.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPNE|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpne.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPNE|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpne.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPNE|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.cmpne.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPNE|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpne.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPNE|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpne.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPNE|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.cmpgt.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGT|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpgt.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGT|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpgt.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPGT|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.cmpgt.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGT|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpgt.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGT|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpgt.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPGT|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.cmpge.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGE|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpge.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGE|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpge.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPGE|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.cmpge.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGE|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpge.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGE|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmpge.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPGE|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.cmplt.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLT|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmplt.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLT|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmplt.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPLT|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.cmplt.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLT|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmplt.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLT|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmplt.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPLT|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.cmple.h",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLE|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmple.sc.h",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLE|MATCH_V_OP_H_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmple.sci.h",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPLE|MATCH_V_OP_H_VI,       MASK_V_OP1,     match_opcode,   0},
{"pv.cmple.b",          0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLE|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.cmple.sc.b",       0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLE|MATCH_V_OP_B_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.cmple.sci.b",      0, INSN_CLASS_XPULP_VECT, "d,s,bs",       MATCH_V_OP_CMPLE|MATCH_V_OP_B_VI,       MASK_V_OP1,     match_opcode,   0},

{"pv.cmpgtu.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGTU|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgtu.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGTU|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgtu.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPGTU|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.cmpgtu.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGTU|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgtu.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGTU|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgtu.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPGTU|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.cmpgeu.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGEU|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgeu.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGEU|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgeu.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPGEU|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.cmpgeu.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGEU|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgeu.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPGEU|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpgeu.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPGEU|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.cmpltu.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLTU|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpltu.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLTU|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpltu.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPLTU|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.cmpltu.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLTU|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpltu.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLTU|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpltu.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPLTU|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.cmpleu.h",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLEU|MATCH_V_OP_H_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpleu.sc.h",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLEU|MATCH_V_OP_H_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpleu.sci.h",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPLEU|MATCH_V_OP_H_VI,      MASK_V_OP1,     match_opcode,   0},
{"pv.cmpleu.b",         0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLEU|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpleu.sc.b",      0, INSN_CLASS_XPULP_VECT, "d,s,t",        MATCH_V_OP_CMPLEU|MATCH_V_OP_B_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.cmpleu.sci.b",     0, INSN_CLASS_XPULP_VECT, "d,s,bu",       MATCH_V_OP_CMPLEU|MATCH_V_OP_B_VI,      MASK_V_OP1,     match_opcode,   0},

{"pv.add.h.div2",       0, INSN_CLASS_XPULP_VECT,   "d,s,t", MATCH_V_OP_ADD_DIV|MATCH_V_OP_H_VV_S1,  MASK_V_OP,      match_opcode,   0},
{"pv.add.h.div4",       0, INSN_CLASS_XPULP_VECT,   "d,s,t", MATCH_V_OP_ADD_DIV|MATCH_V_OP_H_VV_S2,  MASK_V_OP,      match_opcode,   0},
{"pv.add.h.div8",       0, INSN_CLASS_XPULP_VECT,   "d,s,t", MATCH_V_OP_ADD_DIV|MATCH_V_OP_H_VV_S3,  MASK_V_OP,      match_opcode,   0},

{"pv.sub.h.div2",       0, INSN_CLASS_XPULP_VECT,   "d,s,t", MATCH_V_OP_SUB_DIV|MATCH_V_OP_H_VV_S1,  MASK_V_OP,      match_opcode,   0},
{"pv.sub.h.div4",       0, INSN_CLASS_XPULP_VECT,   "d,s,t", MATCH_V_OP_SUB_DIV|MATCH_V_OP_H_VV_S2,  MASK_V_OP,      match_opcode,   0},
{"pv.sub.h.div8",       0, INSN_CLASS_XPULP_VECT,   "d,s,t", MATCH_V_OP_SUB_DIV|MATCH_V_OP_H_VV_S3,  MASK_V_OP,      match_opcode,   0},

/* shuffle pack */
{"pv.shuffle.h",        0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_SHUFFLE|MATCH_V_OP_H_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.shuffle.sci.h",    0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,bu",       MATCH_V_OP_SHUFFLE|MATCH_V_OP_H_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.shuffle.b",        0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_SHUFFLE|MATCH_V_OP_B_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.shufflei0.sci.b",  0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,bu",       MATCH_V_OP_SHUFFLE|MATCH_V_OP_B_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.shufflei1.sci.b",  0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,bu",       MATCH_V_OP_SHUFFLEI1|MATCH_V_OP_B_VI,   MASK_V_OP1,     match_opcode,   0},
{"pv.shufflei2.sci.b",  0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,bu",       MATCH_V_OP_SHUFFLEI2|MATCH_V_OP_B_VI,   MASK_V_OP1,     match_opcode,   0},
{"pv.shufflei3.sci.b",  0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,bu",       MATCH_V_OP_SHUFFLEI3|MATCH_V_OP_B_VI,   MASK_V_OP1,     match_opcode,   0},

{"pv.shuffle2.h",       0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_SHUFFLE2|MATCH_V_OP_H_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.shuffle2.b",       0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_SHUFFLE2|MATCH_V_OP_B_VV,    MASK_V_OP,      match_opcode,   0},

{"pv.pack",             0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_PACK|MATCH_V_OP_H_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.pack.h",           0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_PACKH|MATCH_V_OP_H_VV,        MASK_V_OP,      match_opcode,   0},

{"pv.packhi.b",         0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_PACKHI|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.packlo.b",         0, INSN_CLASS_XPULP_VECT_SHUFFLEPACK, "d,s,t",        MATCH_V_OP_PACKLO|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},

/* PULP branch */
{"p.beqimm",            0, INSN_CLASS_XPULP_BR,  "s,bI,p",      MATCH_BEQM1,                            MASK_BEQM1,     match_opcode,   0},
{"p.bneimm",            0, INSN_CLASS_XPULP_BR,  "s,bI,p",      MATCH_BNEM1,                            MASK_BNEM1,     match_opcode,   0},


/* PULP elw */
/* clust synchronization instruction */
{"p.elw",               0, INSN_CLASS_XPULP_ELW, "d,o(s)",       MATCH_LWU,                              MASK_LWU,       match_opcode,   0},

/* PULP bitrev */
{"p.bitrev",            0, INSN_CLASS_XPULP_BITREV,   "d,s,bi,b5",     MATCH_BITREV,                           MASK_BITREV,    match_opcode,   0},

/* PULP vectcomplex */
{"pv.cplxmul.h.r",      0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S0, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.r.div2", 0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S1, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.r.div4", 0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S2, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.r.div8", 0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S3, MASK_V_OP11,    match_opcode,   0},

{"pv.cplxmul.h.i",      0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S0, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.i.div2", 0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S1, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.i.div4", 0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S2, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.i.div8", 0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S3, MASK_V_OP11,    match_opcode,   0},

{"pv.subrotmj.h",       0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S0, MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div2",  0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S1, MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div4",  0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S2, MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div8",  0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S3, MASK_V_OP,      match_opcode,   0},

{"pv.cplxconj.h",       0, INSN_CLASS_XPULP_VECT_COMPLEX,   "d,s",   MATCH_V_OP_CPLXCONJ|MATCH_V_OP_H_VV,    MASK_V_OP2,     match_opcode,   0},

/* PULP vectgap8 */
/* Vector, Gap8 only */
{"pv.pack.h.h",         0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_PACK|MATCH_V_OP_H_VI,        MASK_V_OP,      match_opcode,   0},
{"pv.pack.l.h",         0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_PACK|MATCH_V_OP_H_VR,        MASK_V_OP,      match_opcode,   0},

{"pv.cplxmul.s",        0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_CPLXMUL|MATCH_V_OP_H_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.cplxmul.sc.s",     0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_CPLXMUL|MATCH_V_OP_H_VR,     MASK_V_OP,      match_opcode,   0},
{"pv.cplxmul.sci.s",    0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,bs",        MATCH_V_OP_CPLXMUL|MATCH_V_OP_H_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.cplxmul.s.div2",   0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_CPLXMUL|MATCH_V_OP_H_VV_PS1, MASK_V_OP,      match_opcode,   0},
{"pv.cplxmul.s.div4",   0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_CPLXMUL|MATCH_V_OP_H_VV_PS2, MASK_V_OP,      match_opcode,   0},

{"pv.subrotmj.h",       0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div2",  0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_PS1,MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div4",  0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_PS2,MASK_V_OP,      match_opcode,   0},

{"pv.add.b.div2",       0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_ADD|MATCH_V_OP_B_VV_PS1,     MASK_V_OP,      match_opcode,   0},
{"pv.add.b.div4",       0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_ADD|MATCH_V_OP_B_VV_PS2,     MASK_V_OP,      match_opcode,   0},

{"pv.sub.b.div2",       0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_SUB|MATCH_V_OP_B_VV_PS1,     MASK_V_OP,      match_opcode,   0},
{"pv.sub.b.div4",       0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_SUB|MATCH_V_OP_B_VV_PS2,     MASK_V_OP,      match_opcode,   0},

{"pv.vitop.max",        0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_VITOP|MATCH_V_OP_B_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.vitop.sel",        0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s,t", MATCH_V_OP_VITOP|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},

{"pv.cplxconj.h",       0, INSN_CLASS_XPULP_VECT_GAP8,   "d,s",   MATCH_V_OP_CPLXCONJ|MATCH_V_OP_H_VV,    MASK_V_OP2,     match_opcode,   0},

/* PULP NN */
/* Xpulpnn vectorial add: nibble, crumb */
{"pv.add.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.add.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.add.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.add.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_ADD|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial sub: nibble, crumb */
{"pv.sub.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sub.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SUB|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial avg: nibble, crumb */
{"pv.avg.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.avg.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVG|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial avgu: nibble, crumb */
{"pv.avgu.n",           0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_N_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.sc.n",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_N_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.c",           0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_C_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.avgu.sc.c",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AVGU|MATCH_V_OP_C_VR,        MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial min: nibble, crumb */
{"pv.min.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.min.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.min.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.min.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MIN|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial minu: nibble, crumb */
{"pv.minu.n",           0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_N_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.sc.n",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_N_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.c",           0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_C_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.minu.sc.c",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MINU|MATCH_V_OP_C_VR,        MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial max: nibble, crumb */
{"pv.max.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.max.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.max.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.max.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAX|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial maxu: nibble, crumb */
{"pv.maxu.n",           0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_N_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.sc.n",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_N_VR,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.c",           0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_C_VV,        MASK_V_OP,      match_opcode,   0},
{"pv.maxu.sc.c",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_MAXU|MATCH_V_OP_C_VR,        MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial srl: nibble, crumb */
{"pv.srl.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.srl.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRL|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial sra: nibble, crumb */
{"pv.sra.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sra.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SRA|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial sll: nibble, crumb */
{"pv.sll.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.sll.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SLL|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial or: nibble, crumb */
{"pv.or.n",             0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_N_VV,          MASK_V_OP,      match_opcode,   0},
{"pv.or.sc.n",          0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_N_VR,          MASK_V_OP,      match_opcode,   0},
{"pv.or.c",             0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_C_VV,          MASK_V_OP,      match_opcode,   0},
{"pv.or.sc.c",          0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_OR|MATCH_V_OP_C_VR,          MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial xor: nibble, crumb */
{"pv.xor.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.xor.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_XOR|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial and: nibble, crumb */
{"pv.and.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_N_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.and.sc.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_N_VR,         MASK_V_OP,      match_opcode,   0},
{"pv.and.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_C_VV,         MASK_V_OP,      match_opcode,   0},
{"pv.and.sc.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_AND|MATCH_V_OP_C_VR,         MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial abs: nibble, crumb */
{"pv.abs.n",            0, INSN_CLASS_XPULP_NN, "d,s",          MATCH_V_OP_ABS|MATCH_V_OP_N_VV,         MASK_V_OP2,     match_opcode,   0},
{"pv.abs.c",            0, INSN_CLASS_XPULP_NN, "d,s",          MATCH_V_OP_ABS|MATCH_V_OP_C_VV,         MASK_V_OP2,     match_opcode,   0},
/* Xpulpnn vectorial dotup: nibble, crumb */
{"pv.dotup.n",          0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_N_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.sc.n",       0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_N_VR,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.c",          0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_C_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.dotup.sc.c",       0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_EXTRACTU|MATCH_V_OP_C_VR,    MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial dotusp: nibble, crumb */
{"pv.dotusp.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_N_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.sc.n",      0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_N_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_C_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.dotusp.sc.c",      0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_INSERT|MATCH_V_OP_C_VR,      MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial dotsp: nibble, crumb */
{"pv.dotsp.n",          0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_N_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.sc.n",       0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_N_VR,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.c",          0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_C_VV,       MASK_V_OP,      match_opcode,   0},
{"pv.dotsp.sc.c",       0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUP|MATCH_V_OP_C_VR,       MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial sdotup: nibble, crumb */
{"pv.sdotup.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_N_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.sc.n",      0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_N_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_C_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotup.sc.c",      0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_DOTUSP|MATCH_V_OP_C_VR,      MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial sdotusp: nibble, crumb */
{"pv.sdotusp.n",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_N_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.sc.n",     0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_N_VR,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.c",        0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_C_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.sdotusp.sc.c",     0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTSP|MATCH_V_OP_C_VR,      MASK_V_OP,      match_opcode,   0},
/* Xpulpnn vectorial sdotsp: nibble, crumb */
{"pv.sdotsp.n",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_N_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.sc.n",      0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_N_VR,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.c",         0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_C_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.sdotsp.sc.c",      0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_SDOTUSP|MATCH_V_OP_C_VR,     MASK_V_OP,      match_opcode,   0},

/* Xpulpnn quantization: nibble, crumble */
{"pv.qnt.n",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_PACKLO|MATCH_V_OP_N_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.qnt.c",            0, INSN_CLASS_XPULP_NN, "d,s,t",        MATCH_V_OP_PACKLO|MATCH_V_OP_C_VV,      MASK_V_OP,      match_opcode,   0},

/* To be integrated for nibble&crumb somhow -- needed more encoding space -- talk to Giuseppe about it

{"pv.shuffle.h",        "Xpulpv3", "d,s,t",     MATCH_V_OP_SHUFFLE|MATCH_V_OP_H_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.shuffle.sci.h",    "Xpulpv3", "d,s,bu",    MATCH_V_OP_SHUFFLE|MATCH_V_OP_H_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.shuffle.b",        "Xpulpv3", "d,s,t",     MATCH_V_OP_SHUFFLE|MATCH_V_OP_B_VV,     MASK_V_OP,      match_opcode,   0},
{"pv.shufflei0.sci.b",  "Xpulpv3", "d,s,bu",    MATCH_V_OP_SHUFFLE|MATCH_V_OP_B_VI,     MASK_V_OP1,     match_opcode,   0},
{"pv.shufflei1.sci.b",  "Xpulpv3", "d,s,bu",    MATCH_V_OP_SHUFFLEI1|MATCH_V_OP_B_VI,   MASK_V_OP1,     match_opcode,   0},
{"pv.shufflei2.sci.b",  "Xpulpv3", "d,s,bu",    MATCH_V_OP_SHUFFLEI2|MATCH_V_OP_B_VI,   MASK_V_OP1,     match_opcode,   0},
{"pv.shufflei3.sci.b",  "Xpulpv3", "d,s,bu",    MATCH_V_OP_SHUFFLEI3|MATCH_V_OP_B_VI,   MASK_V_OP1,     match_opcode,   0},

{"pv.shuffle2.h",       "Xpulpv3", "d,s,t",     MATCH_V_OP_SHUFFLE2|MATCH_V_OP_H_VV,    MASK_V_OP,      match_opcode,   0},
{"pv.shuffle2.b",       "Xpulpv3", "d,s,t",     MATCH_V_OP_SHUFFLE2|MATCH_V_OP_B_VV,    MASK_V_OP,      match_opcode,   0},

{"pv.pack.h",           "Xpulpv3", "d,s,t",     MATCH_V_OP_PACK|MATCH_V_OP_H_VV,        MASK_V_OP,      match_opcode,   0},

{"pv.packhi.b",                 "Xpulpv3", "d,s,t",     MATCH_V_OP_PACKHI|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0},
{"pv.packlo.b",                 "Xpulpv3", "d,s,t",     MATCH_V_OP_PACKLO|MATCH_V_OP_B_VV,      MASK_V_OP,      match_opcode,   0}, */

/* Xpulpnn extensions v2 */
{"pv.mlsdotup.h.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP|MATCH_V_OP_ML_H_VV,   MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotup.h.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_1|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotup.h.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_2|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.h.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_3|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.b.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP|MATCH_V_OP_ML_B_VV, MASK_V_OP,    match_opcode, 0},
{"pv.mlsdotup.b.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_1|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.b.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_2|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.b.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_3|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.n.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP|MATCH_V_OP_ML_N_VV,   MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotup.n.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_1|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotup.n.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_2|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.n.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_3|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.c.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP|MATCH_V_OP_ML_C_VV, MASK_V_OP,    match_opcode, 0},
{"pv.mlsdotup.c.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_1|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.c.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_2|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotup.c.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_DOTUSP_3|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},

{"pv.mlsdotusp.h.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP|MATCH_V_OP_ML_H_VV,   MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.h.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_1|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.h.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_2|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.h.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_3|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.b.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP|MATCH_V_OP_ML_B_VV,   MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.b.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_1|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.b.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_2|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.b.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_3|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.n.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP|MATCH_V_OP_ML_N_VV,   MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.n.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_1|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.n.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_2|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.n.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_3|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.c.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP|MATCH_V_OP_ML_C_VV,   MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.c.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_1|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotusp.c.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_2|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotusp.c.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTSP_3|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},

{"pv.mlsdotsup.h.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP|MATCH_V_OP_ML_H_VV, MASK_V_OP,    match_opcode, 0},
{"pv.mlsdotsup.h.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_1|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.h.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_2|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.h.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_3|MATCH_V_OP_ML_H_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.b.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP|MATCH_V_OP_ML_B_VV, MASK_V_OP,    match_opcode, 0},
{"pv.mlsdotsup.b.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_1|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.b.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_2|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.b.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_3|MATCH_V_OP_ML_B_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.n.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP|MATCH_V_OP_ML_N_VV, MASK_V_OP,    match_opcode, 0},
{"pv.mlsdotsup.n.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_1|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.n.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_2|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.n.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_3|MATCH_V_OP_ML_N_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.c.0",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP|MATCH_V_OP_ML_C_VV, MASK_V_OP,    match_opcode, 0},
{"pv.mlsdotsup.c.1",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_1|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.c.2",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_2|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsup.c.3",  0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_ML_SDOTSUP_3|MATCH_V_OP_ML_C_VV, MASK_V_OP,  match_opcode, 0},

{"pv.mlsdotsp.h.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP|MATCH_V_OP_ML_H_VV,    MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.h.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_1|MATCH_V_OP_ML_H_VV,  MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.h.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_2|MATCH_V_OP_ML_H_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.h.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_3|MATCH_V_OP_ML_H_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.b.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP|MATCH_V_OP_ML_B_VV,    MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.b.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_1|MATCH_V_OP_ML_B_VV,  MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.b.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_2|MATCH_V_OP_ML_B_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.b.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_3|MATCH_V_OP_ML_B_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.n.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP|MATCH_V_OP_ML_N_VV,    MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.n.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_1|MATCH_V_OP_ML_N_VV,  MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.n.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_2|MATCH_V_OP_ML_N_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.n.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_3|MATCH_V_OP_ML_N_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.c.0",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP|MATCH_V_OP_ML_C_VV,    MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.c.1",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_1|MATCH_V_OP_ML_C_VV,  MASK_V_OP,  match_opcode,   0},
{"pv.mlsdotsp.c.2",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_2|MATCH_V_OP_ML_C_VV,  MASK_V_OP,  match_opcode, 0},
{"pv.mlsdotsp.c.3",   0, INSN_CLASS_XPULP_NN, "d,s,t", MATCH_V_OP_SDOTUSP_3|MATCH_V_OP_ML_C_VV,  MASK_V_OP,  match_opcode, 0},

/*
{"p.avgu",           0, INSN_CLASS_XPULP_V3, "d,s,t",        MATCH_AVGU,                             MASK_PALU,      match_opcode,   0},
{"pv.ball",             0, INSN_CLASS_XPULP_V3,  "s,p",         MATCH_BEQM1,                    MASK_BEQM1|MASK_RS2,    match_opcode,   },
{"pv.bnall",            0, INSN_CLASS_XPULP_V3,  "s,p",         MATCH_BNEM1,                    MASK_BNEM1|MASK_RS2,    match_opcode,   },
*/
/* TODO: hack, remove */
#define WR_xd 0
#define WR_fd 0
#define RD_xs1 0
#define RD_xs2 0
#define RD_xs3 0
#define RD_fs1 0
#define RD_fs2 0
#define RD_fs3 0

/* Floating point extensions: half/quarter floats and vector instructions */
/* RV32Xfhalf - half-precision floats */
{"flh",       0, INSN_CLASS_XPULP_FHALF, "D,o(s)",  MATCH_FLH, MASK_FLH, match_opcode,   WR_xd|RD_xs1 },
{"flh",       0, INSN_CLASS_XPULP_FHALF, "D,A,s",  0, (int) M_FLH, match_never,  INSN_MACRO },
{"fsh",       0, INSN_CLASS_XPULP_FHALF, "T,q(s)",  MATCH_FSH, MASK_FSH, match_opcode,   RD_xs1|RD_xs2 },
{"fsh",       0, INSN_CLASS_XPULP_FHALF, "T,A,s",  0, (int) M_FSH, match_never,  INSN_MACRO },
{"fmadd.h",   0, INSN_CLASS_XPULP_FHALF, "D,S,T,R",  MATCH_FMADD_H | MASK_RM, MASK_FMADD_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmadd.h",   0, INSN_CLASS_XPULP_FHALF, "D,S,T,R,m",  MATCH_FMADD_H, MASK_FMADD_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.h",   0, INSN_CLASS_XPULP_FHALF, "D,S,T,R",  MATCH_FMSUB_H | MASK_RM, MASK_FMSUB_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.h",   0, INSN_CLASS_XPULP_FHALF, "D,S,T,R,m",  MATCH_FMSUB_H, MASK_FMSUB_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.h",  0, INSN_CLASS_XPULP_FHALF, "D,S,T,R",  MATCH_FNMSUB_H | MASK_RM, MASK_FNMSUB_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.h",  0, INSN_CLASS_XPULP_FHALF, "D,S,T,R,m",  MATCH_FNMSUB_H, MASK_FNMSUB_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.h",  0, INSN_CLASS_XPULP_FHALF, "D,S,T,R",  MATCH_FNMADD_H | MASK_RM, MASK_FNMADD_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.h",  0, INSN_CLASS_XPULP_FHALF, "D,S,T,R,m",  MATCH_FNMADD_H, MASK_FNMADD_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fadd.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FADD_H | MASK_RM, MASK_FADD_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fadd.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T,m",  MATCH_FADD_H, MASK_FADD_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FSUB_H | MASK_RM, MASK_FSUB_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T,m",  MATCH_FSUB_H, MASK_FSUB_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FMUL_H | MASK_RM, MASK_FMUL_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T,m",  MATCH_FMUL_H, MASK_FMUL_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FDIV_H | MASK_RM, MASK_FDIV_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T,m",  MATCH_FDIV_H, MASK_FDIV_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsqrt.h",   0, INSN_CLASS_XPULP_FHALF, "D,S",  MATCH_FSQRT_H | MASK_RM, MASK_FSQRT_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fsqrt.h",   0, INSN_CLASS_XPULP_FHALF, "D,S,m",  MATCH_FSQRT_H, MASK_FSQRT_H, match_opcode,  WR_xd|RD_xs1 },
{"fsgnj.h",   0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FSGNJ_H, MASK_FSGNJ_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjn.h",  0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FSGNJN_H, MASK_FSGNJN_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjx.h",  0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FSGNJX_H, MASK_FSGNJX_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmin.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FMIN_H, MASK_FMIN_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmax.h",    0, INSN_CLASS_XPULP_FHALF, "D,S,T",  MATCH_FMAX_H, MASK_FMAX_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"feq.h",     0, INSN_CLASS_XPULP_FHALF, "d,S,T",    MATCH_FEQ_H, MASK_FEQ_H, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"flt.h",     0, INSN_CLASS_XPULP_FHALF, "d,S,T",    MATCH_FLT_H, MASK_FLT_H, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fle.h",     0, INSN_CLASS_XPULP_FHALF, "d,S,T",    MATCH_FLE_H, MASK_FLE_H, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fcvt.w.h",  0, INSN_CLASS_XPULP_FHALF, "d,S",  MATCH_FCVT_W_H | MASK_RM, MASK_FCVT_W_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.w.h",  0, INSN_CLASS_XPULP_FHALF, "d,S,m",  MATCH_FCVT_W_H, MASK_FCVT_W_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.h", 0, INSN_CLASS_XPULP_FHALF, "d,S",  MATCH_FCVT_WU_H | MASK_RM, MASK_FCVT_WU_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.h", 0, INSN_CLASS_XPULP_FHALF, "d,S,m",  MATCH_FCVT_WU_H, MASK_FCVT_WU_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.h.w",  0, INSN_CLASS_XPULP_FHALF, "D,s",  MATCH_FCVT_H_W | MASK_RM, MASK_FCVT_H_W | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.w",  0, INSN_CLASS_XPULP_FHALF, "D,s,m",  MATCH_FCVT_H_W, MASK_FCVT_H_W, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.wu", 0, INSN_CLASS_XPULP_FHALF, "D,s",  MATCH_FCVT_H_WU | MASK_RM, MASK_FCVT_H_W | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.wu", 0, INSN_CLASS_XPULP_FHALF, "D,s,m",  MATCH_FCVT_H_WU, MASK_FCVT_H_WU, match_opcode,   WR_xd|RD_xs1 },
{"fmv.x.h",   0, INSN_CLASS_XPULP_FHALF, "d,S",  MATCH_FMV_X_H, MASK_FMV_X_H, match_opcode,  WR_xd|RD_xs1 },
{"fclass.h",  0, INSN_CLASS_XPULP_FHALF, "d,S",  MATCH_FCLASS_H, MASK_FCLASS_H, match_opcode,   WR_xd|RD_xs1 },
{"fmv.h.x",   0, INSN_CLASS_XPULP_FHALF, "D,s",  MATCH_FMV_H_X, MASK_FMV_H_X, match_opcode,  WR_xd|RD_xs1 },
/* pseudos */
{"fgt.h",     0, INSN_CLASS_XPULP_FHALF, "d,T,S",    MATCH_FLT_H, MASK_FLT_H, match_opcode,  INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fge.h",     0, INSN_CLASS_XPULP_FHALF, "d,T,S",    MATCH_FLE_H, MASK_FLE_H, match_opcode,  INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fmv.h",     0, INSN_CLASS_XPULP_FHALF, "D,U",  MATCH_FSGNJ_H, MASK_FSGNJ_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fabs.h",    0, INSN_CLASS_XPULP_FHALF, "D,U",  MATCH_FSGNJX_H, MASK_FSGNJX_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fneg.h",    0, INSN_CLASS_XPULP_FHALF, "D,U",  MATCH_FSGNJN_H, MASK_FSGNJN_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },

/* RV32Xfhalfinx - half-precision floats in x registers (no flh/fsh/fmv/)*/
{"fmadd.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r",  MATCH_FMADD_H | MASK_RM, MASK_FMADD_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmadd.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r,m",  MATCH_FMADD_H, MASK_FMADD_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r",  MATCH_FMSUB_H | MASK_RM, MASK_FMSUB_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r,m",  MATCH_FMSUB_H, MASK_FMSUB_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r",  MATCH_FNMSUB_H | MASK_RM, MASK_FNMSUB_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r,m",  MATCH_FNMSUB_H, MASK_FNMSUB_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r",  MATCH_FNMADD_H | MASK_RM, MASK_FNMADD_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,r,m",  MATCH_FNMADD_H, MASK_FNMADD_H, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fadd.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FADD_H | MASK_RM, MASK_FADD_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fadd.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,m",  MATCH_FADD_H, MASK_FADD_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FSUB_H | MASK_RM, MASK_FSUB_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,m",  MATCH_FSUB_H, MASK_FSUB_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FMUL_H | MASK_RM, MASK_FMUL_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,m",  MATCH_FMUL_H, MASK_FMUL_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FDIV_H | MASK_RM, MASK_FDIV_H | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t,m",  MATCH_FDIV_H, MASK_FDIV_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsqrt.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FSQRT_H | MASK_RM, MASK_FSQRT_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fsqrt.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FSQRT_H, MASK_FSQRT_H, match_opcode,  WR_xd|RD_xs1 },
{"fsgnj.h",   0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FSGNJ_H, MASK_FSGNJ_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjn.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FSGNJN_H, MASK_FSGNJN_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjx.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FSGNJX_H, MASK_FSGNJX_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmin.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FMIN_H, MASK_FMIN_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmax.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",  MATCH_FMAX_H, MASK_FMAX_H, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"feq.h",     0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",    MATCH_FEQ_H, MASK_FEQ_H, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"flt.h",     0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",    MATCH_FLT_H, MASK_FLT_H, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fle.h",     0, INSN_CLASS_XPULP_FHALFINX, "d,s,t",    MATCH_FLE_H, MASK_FLE_H, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fcvt.w.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_W_H | MASK_RM, MASK_FCVT_W_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.w.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_W_H, MASK_FCVT_W_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.h", 0, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_WU_H | MASK_RM, MASK_FCVT_WU_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.h", 0, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_WU_H, MASK_FCVT_WU_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.h.w",  0, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_H_W | MASK_RM, MASK_FCVT_H_W | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.w",  0, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_H_W, MASK_FCVT_H_W, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.wu", 0, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_H_WU | MASK_RM, MASK_FCVT_H_W | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.wu", 0, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_H_WU, MASK_FCVT_H_WU, match_opcode,   WR_xd|RD_xs1 },
{"fclass.h",  0, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCLASS_H, MASK_FCLASS_H, match_opcode,   WR_xd|RD_xs1 },
/* pseudos */
{"fgt.h",     0, INSN_CLASS_XPULP_FHALFINX, "d,t,s",    MATCH_FLT_H, MASK_FLT_H, match_opcode,  INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fge.h",     0, INSN_CLASS_XPULP_FHALFINX, "d,t,s",    MATCH_FLE_H, MASK_FLE_H, match_opcode,  INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fabs.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,u",  MATCH_FSGNJX_H, MASK_FSGNJX_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fneg.h",    0, INSN_CLASS_XPULP_FHALFINX, "d,u",  MATCH_FSGNJN_H, MASK_FSGNJN_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },

/* RV64Xfhalf Half-Precision Floating-Point Extension (in addition to RV32Xfhalf)*/
{"fcvt.l.h",  64, INSN_CLASS_XPULP_FHALF, "d,S",  MATCH_FCVT_L_H | MASK_RM, MASK_FCVT_L_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.l.h",  64, INSN_CLASS_XPULP_FHALF, "d,S,m",  MATCH_FCVT_L_H, MASK_FCVT_L_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.h", 64, INSN_CLASS_XPULP_FHALF, "d,S",  MATCH_FCVT_LU_H | MASK_RM, MASK_FCVT_LU_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.h", 64, INSN_CLASS_XPULP_FHALF, "d,S,m",  MATCH_FCVT_LU_H, MASK_FCVT_LU_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.h.l",  64, INSN_CLASS_XPULP_FHALF, "D,s",  MATCH_FCVT_H_L | MASK_RM, MASK_FCVT_H_L | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.l",  64, INSN_CLASS_XPULP_FHALF, "D,s,m",  MATCH_FCVT_H_L, MASK_FCVT_H_L, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.lu", 64, INSN_CLASS_XPULP_FHALF, "D,s",  MATCH_FCVT_H_LU | MASK_RM, MASK_FCVT_H_L | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.lu", 64, INSN_CLASS_XPULP_FHALF, "D,s,m",  MATCH_FCVT_H_LU, MASK_FCVT_H_LU, match_opcode,   WR_xd|RD_xs1 },

/* RV64Xfhalfinx Half-Precision Floating-Point Extension (in addition to RV32Xfhalfinx) in x registers */
{"fcvt.l.h",  64, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_L_H | MASK_RM, MASK_FCVT_L_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.l.h",  64, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_L_H, MASK_FCVT_L_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.h", 64, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_LU_H | MASK_RM, MASK_FCVT_LU_H | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.h", 64, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_LU_H, MASK_FCVT_LU_H, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.h.l",  64, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_H_L | MASK_RM, MASK_FCVT_H_L | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.l",  64, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_H_L, MASK_FCVT_H_L, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.lu", 64, INSN_CLASS_XPULP_FHALFINX, "d,s",  MATCH_FCVT_H_LU | MASK_RM, MASK_FCVT_H_L | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.lu", 64, INSN_CLASS_XPULP_FHALFINX, "d,s,m",  MATCH_FCVT_H_LU, MASK_FCVT_H_LU, match_opcode,   WR_xd|RD_xs1 },

/* XfhalfwithF - half-precision conversions with F extension */
{"fcvt.s.h",  0, INSN_CLASS_XPULP_FHALFWITHF, "D,S",  MATCH_FCVT_S_H, MASK_FCVT_S_H, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.h.s",  0, INSN_CLASS_XPULP_FHALFWITHF, "D,S",  MATCH_FCVT_H_S | MASK_RM, MASK_FCVT_H_S | MASK_RM, match_opcode,   WR_xd|RD_fs1 },
{"fcvt.h.s",  0, INSN_CLASS_XPULP_FHALFWITHF, "D,S,m",  MATCH_FCVT_H_S, MASK_FCVT_H_S, match_opcode,   WR_xd|RD_fs1 },

/* XfhalfwithFinx - half-precision conversions with F extension in x registers */
{"fcvt.s.h",  0, INSN_CLASS_XPULP_FHALFWITHFINX, "d,s",  MATCH_FCVT_S_H, MASK_FCVT_S_H, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.h.s",  0, INSN_CLASS_XPULP_FHALFWITHFINX, "d,s",  MATCH_FCVT_H_S | MASK_RM, MASK_FCVT_H_S | MASK_RM, match_opcode,   WR_xd|RD_fs1 },
{"fcvt.h.s",  0, INSN_CLASS_XPULP_FHALFWITHFINX, "d,s,m",  MATCH_FCVT_H_S, MASK_FCVT_H_S, match_opcode,   WR_xd|RD_fs1 },

/* XfhalfwithD - half-precision conversions with D extension (in addition to XfhalfwithF) */
{"fcvt.d.h",  0, INSN_CLASS_XPULP_FHALFWITHD, "D,S",  MATCH_FCVT_D_H, MASK_FCVT_D_H, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.h.d",  0, INSN_CLASS_XPULP_FHALFWITHD, "D,S",  MATCH_FCVT_H_D | MASK_RM, MASK_FCVT_H_D | MASK_RM, match_opcode,   WR_xd|RD_fs1 },
{"fcvt.h.d",  0, INSN_CLASS_XPULP_FHALFWITHD, "D,S,m",  MATCH_FCVT_H_D, MASK_FCVT_H_D, match_opcode,   WR_xd|RD_fs1 },

/* XfhalfwithDinx - half-precision conversions with D extension (in addition to XfhalfwithF) in x registers*/
{"fcvt.d.h",  0, INSN_CLASS_XPULP_FHALFWITHDINX, "d,s",  MATCH_FCVT_D_H, MASK_FCVT_D_H, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.h.d",  0, INSN_CLASS_XPULP_FHALFWITHDINX, "d,s",  MATCH_FCVT_H_D | MASK_RM, MASK_FCVT_H_D | MASK_RM, match_opcode,   WR_xd|RD_fs1 },
{"fcvt.h.d",  0, INSN_CLASS_XPULP_FHALFWITHDINX, "d,s,m",  MATCH_FCVT_H_D, MASK_FCVT_H_D, match_opcode,   WR_xd|RD_fs1 },

/* RV32Xfalthalf - alternate half-precision floats */
{"flah",       0, INSN_CLASS_XPULP_FALTHALF, "D,o(s)",  MATCH_FLH, MASK_FLH, match_opcode,   INSN_ALIAS|WR_xd|RD_xs1 },
{"flah",       0, INSN_CLASS_XPULP_FALTHALF, "D,A,s",  0, (int) M_FLH, match_never,  INSN_MACRO },
{"fsah",       0, INSN_CLASS_XPULP_FALTHALF, "T,q(s)",  MATCH_FSH, MASK_FSH, match_opcode,   INSN_ALIAS|RD_xs1|RD_xs2 },
{"fsah",       0, INSN_CLASS_XPULP_FALTHALF, "T,A,s",  0, (int) M_FSH, match_never,  INSN_MACRO },
{"fmadd.ah",   0, INSN_CLASS_XPULP_FALTHALF, "D,S,T,R",  MATCH_FMADD_AH, MASK_FMADD_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.ah",   0, INSN_CLASS_XPULP_FALTHALF, "D,S,T,R",  MATCH_FMSUB_AH, MASK_FMSUB_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.ah",  0, INSN_CLASS_XPULP_FALTHALF, "D,S,T,R",  MATCH_FNMSUB_AH, MASK_FNMSUB_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.ah",  0, INSN_CLASS_XPULP_FALTHALF, "D,S,T,R",  MATCH_FNMADD_AH, MASK_FNMADD_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fadd.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FADD_AH, MASK_FADD_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FSUB_AH, MASK_FSUB_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FMUL_AH, MASK_FMUL_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FDIV_AH, MASK_FDIV_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsqrt.ah",   0, INSN_CLASS_XPULP_FALTHALF, "D,S",  MATCH_FSQRT_AH, MASK_FSQRT_AH, match_opcode,  WR_xd|RD_xs1 },
{"fsgnj.ah",   0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FSGNJ_AH, MASK_FSGNJ_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjn.ah",  0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FSGNJN_AH, MASK_FSGNJN_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjx.ah",  0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FSGNJX_AH, MASK_FSGNJX_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmin.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FMIN_AH, MASK_FMIN_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmax.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,S,T",  MATCH_FMAX_AH, MASK_FMAX_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"feq.ah",     0, INSN_CLASS_XPULP_FALTHALF, "d,S,T",    MATCH_FEQ_AH, MASK_FEQ_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"flt.ah",     0, INSN_CLASS_XPULP_FALTHALF, "d,S,T",    MATCH_FLT_AH, MASK_FLT_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fle.ah",     0, INSN_CLASS_XPULP_FALTHALF, "d,S,T",    MATCH_FLE_AH, MASK_FLE_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fcvt.w.ah",  0, INSN_CLASS_XPULP_FALTHALF, "d,S",  MATCH_FCVT_W_AH, MASK_FCVT_W_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.ah", 0, INSN_CLASS_XPULP_FALTHALF, "d,S",  MATCH_FCVT_WU_AH, MASK_FCVT_WU_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.ah.w",  0, INSN_CLASS_XPULP_FALTHALF, "D,s",  MATCH_FCVT_AH_W, MASK_FCVT_AH_W, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.ah.wu", 0, INSN_CLASS_XPULP_FALTHALF, "D,s",  MATCH_FCVT_AH_WU, MASK_FCVT_AH_W, match_opcode,   WR_xd|RD_xs1 },
{"fmv.x.ah",   0, INSN_CLASS_XPULP_FALTHALF, "d,S",  MATCH_FMV_X_AH, MASK_FMV_X_AH, match_opcode,  WR_xd|RD_xs1 },
{"fclass.ah",  0, INSN_CLASS_XPULP_FALTHALF, "D,S",  MATCH_FCLASS_AH, MASK_FCLASS_AH, match_opcode,   WR_xd|RD_xs1 },
{"fmv.ah.x",   0, INSN_CLASS_XPULP_FALTHALF, "D,s",  MATCH_FMV_AH_X, MASK_FMV_AH_X, match_opcode,  WR_xd|RD_xs1 },
/* pseudos */
{"fgt.ah",     0, INSN_CLASS_XPULP_FALTHALF, "d,T,S",    MATCH_FLT_AH, MASK_FLT_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fge.ah",     0, INSN_CLASS_XPULP_FALTHALF, "d,T,S",    MATCH_FLE_AH, MASK_FLE_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fmv.ah",     0, INSN_CLASS_XPULP_FALTHALF, "D,U",  MATCH_FSGNJ_AH, MASK_FSGNJ_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fabs.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,U",  MATCH_FSGNJX_AH, MASK_FSGNJX_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fneg.ah",    0, INSN_CLASS_XPULP_FALTHALF, "D,U",  MATCH_FSGNJN_AH, MASK_FSGNJN_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },

/* RV32Xfalthalfinx - alternate half-precision floats in x registers (no flah/fsah/fmv/) */
{"fmadd.ah",   0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t,r",  MATCH_FMADD_AH, MASK_FMADD_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.ah",   0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t,r",  MATCH_FMSUB_AH, MASK_FMSUB_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.ah",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t,r",  MATCH_FNMSUB_AH, MASK_FNMSUB_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.ah",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t,r",  MATCH_FNMADD_AH, MASK_FNMADD_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fadd.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FADD_AH, MASK_FADD_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FSUB_AH, MASK_FSUB_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FMUL_AH, MASK_FMUL_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FDIV_AH, MASK_FDIV_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsqrt.ah",   0, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FSQRT_AH, MASK_FSQRT_AH, match_opcode,  WR_xd|RD_xs1 },
{"fsgnj.ah",   0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FSGNJ_AH, MASK_FSGNJ_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjn.ah",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FSGNJN_AH, MASK_FSGNJN_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjx.ah",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FSGNJX_AH, MASK_FSGNJX_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmin.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FMIN_AH, MASK_FMIN_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmax.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",  MATCH_FMAX_AH, MASK_FMAX_AH, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"feq.ah",     0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",    MATCH_FEQ_AH, MASK_FEQ_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"flt.ah",     0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",    MATCH_FLT_AH, MASK_FLT_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fle.ah",     0, INSN_CLASS_XPULP_FALTHALFINX, "d,s,t",    MATCH_FLE_AH, MASK_FLE_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fcvt.w.ah",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_W_AH, MASK_FCVT_W_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.ah", 0, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_WU_AH, MASK_FCVT_WU_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.ah.w",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_AH_W, MASK_FCVT_AH_W, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.ah.wu", 0, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_AH_WU, MASK_FCVT_AH_W, match_opcode,   WR_xd|RD_xs1 },
{"fclass.ah",  0, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCLASS_AH, MASK_FCLASS_AH, match_opcode,   WR_xd|RD_xs1 },
/* pseudos */
{"fgt.ah",     0, INSN_CLASS_XPULP_FALTHALFINX, "d,t,s",    MATCH_FLT_AH, MASK_FLT_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fge.ah",     0, INSN_CLASS_XPULP_FALTHALFINX, "d,t,s",    MATCH_FLE_AH, MASK_FLE_AH, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fabs.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,u",  MATCH_FSGNJX_AH, MASK_FSGNJX_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fneg.ah",    0, INSN_CLASS_XPULP_FALTHALFINX, "d,u",  MATCH_FSGNJN_AH, MASK_FSGNJN_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },

/* RV64Xfalthalf Alternative Half-Precision Floating-Point Extension (in addition to RV32Xf16)*/
{"fcvt.l.ah",  64, INSN_CLASS_XPULP_FALTHALF, "D,S",  MATCH_FCVT_L_AH, MASK_FCVT_L_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.ah", 64, INSN_CLASS_XPULP_FALTHALF, "D,S",  MATCH_FCVT_LU_AH, MASK_FCVT_LU_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.ah.l",  64, INSN_CLASS_XPULP_FALTHALF, "D,S",  MATCH_FCVT_AH_L, MASK_FCVT_AH_L, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.ah.lu", 64, INSN_CLASS_XPULP_FALTHALF, "D,S",  MATCH_FCVT_AH_LU, MASK_FCVT_AH_L, match_opcode,   WR_xd|RD_xs1 },

/* RV64Xfalthalfinx Alternative Half-Precision Floating-Point Extension (in addition to rv32Xfalthalfinx) in x register */
{"fcvt.l.ah",  64, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_L_AH, MASK_FCVT_L_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.ah", 64, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_LU_AH, MASK_FCVT_LU_AH, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.ah.l",  64, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_AH_L, MASK_FCVT_AH_L, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.ah.lu", 64, INSN_CLASS_XPULP_FALTHALFINX, "d,s",  MATCH_FCVT_AH_LU, MASK_FCVT_AH_L, match_opcode,   WR_xd|RD_xs1 },

/* XfalthalfwithF - alternate half-precision conversions with F extension */
{"fcvt.s.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHF, "D,S",  MATCH_FCVT_S_AH, MASK_FCVT_S_AH, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.ah.s",  0, INSN_CLASS_XPULP_FALTHALFWITHF, "D,S",  MATCH_FCVT_AH_S, MASK_FCVT_AH_S, match_opcode,   WR_xd|RD_fs1 },

/* XfalthalfwithFinx - alternate half-precision conversions with F extension in x registers */
{"fcvt.s.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHFINX, "d,s",  MATCH_FCVT_S_AH, MASK_FCVT_S_AH, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.ah.s",  0, INSN_CLASS_XPULP_FALTHALFWITHFINX, "d,s",  MATCH_FCVT_AH_S, MASK_FCVT_AH_S, match_opcode,   WR_xd|RD_fs1 },

/* XfalthalfwithD - alternate half-precision conversions with D extension (in addition to XfalthalfwithF) */
{"fcvt.d.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHD, "D,S",  MATCH_FCVT_D_AH, MASK_FCVT_D_AH, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.ah.d",  0, INSN_CLASS_XPULP_FALTHALFWITHD, "D,S",  MATCH_FCVT_AH_D, MASK_FCVT_AH_D, match_opcode,   WR_xd|RD_fs1 },

/* XfalthalfwithDinx - alternate half-precision conversions with D extension (in addition to XfalthalfwithF) in x registers */
{"fcvt.d.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHDINX, "d,s",  MATCH_FCVT_D_AH, MASK_FCVT_D_AH, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.ah.d",  0, INSN_CLASS_XPULP_FALTHALFWITHDINX, "d,s",  MATCH_FCVT_AH_D, MASK_FCVT_AH_D, match_opcode,   WR_xd|RD_fs1 },

/* Xfalthalfwithhalf - alternate half-precision conversions with Xfhalf extension */
{"fcvt.h.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHHALF, "D,S"  ,  MATCH_FCVT_H_AH | MASK_RM, MASK_FCVT_H_AH | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHHALF, "D,S,m",  MATCH_FCVT_H_AH, MASK_FCVT_H_AH, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.ah.h",  0, INSN_CLASS_XPULP_FALTHALFWITHHALF, "D,S",    MATCH_FCVT_AH_H, MASK_FCVT_AH_H, match_opcode,   WR_xd|RD_xs1 },

/* Xfalthalfwithhalfinx - alternate half-precision conversions with Xfhalf extension in x registers */
{"fcvt.h.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHHALFINX, "d,s"  ,  MATCH_FCVT_H_AH | MASK_RM, MASK_FCVT_H_AH | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.h.ah",  0, INSN_CLASS_XPULP_FALTHALFWITHHALFINX, "d,s,m",  MATCH_FCVT_H_AH, MASK_FCVT_H_AH, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.ah.h",  0, INSN_CLASS_XPULP_FALTHALFWITHHALFINX, "d,s",    MATCH_FCVT_AH_H, MASK_FCVT_AH_H, match_opcode,   WR_xd|RD_xs1 },

/* RV32Xfquarter - quarter-precision floats */
/* collides with quad precision floats (q) */
{"flb",       0, INSN_CLASS_XPULP_FQUARTER, "D,o(s)",  MATCH_FLB, MASK_FLB, match_opcode,   WR_xd|RD_xs1 },
{"flb",       0, INSN_CLASS_XPULP_FQUARTER, "D,A,s",  0, (int) M_FLB, match_never,  INSN_MACRO },
{"fsb",       0, INSN_CLASS_XPULP_FQUARTER, "T,q(s)",  MATCH_FSB, MASK_FSB, match_opcode,   RD_xs1|RD_xs2 },
{"fsb",       0, INSN_CLASS_XPULP_FQUARTER, "T,A,s",  0, (int) M_FSB, match_never,  INSN_MACRO },
{"fmadd.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R",  MATCH_FMADD_B | MASK_RM, MASK_FMADD_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmadd.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R,m",  MATCH_FMADD_B, MASK_FMADD_B, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R",  MATCH_FMSUB_B | MASK_RM, MASK_FMSUB_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fmsub.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R,m",  MATCH_FMSUB_B, MASK_FMSUB_B, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R",  MATCH_FNMSUB_B | MASK_RM, MASK_FNMSUB_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmsub.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R,m",  MATCH_FNMSUB_B, MASK_FNMSUB_B, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R",  MATCH_FNMADD_B | MASK_RM, MASK_FNMADD_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fnmadd.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,R,m",  MATCH_FNMADD_B, MASK_FNMADD_B, match_opcode,   WR_xd|RD_xs1|RD_xs2|RD_xs3 },
{"fadd.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FADD_B | MASK_RM, MASK_FADD_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fadd.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,m",  MATCH_FADD_B, MASK_FADD_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FSUB_B | MASK_RM, MASK_FSUB_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsub.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,m",  MATCH_FSUB_B, MASK_FSUB_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FMUL_B | MASK_RM, MASK_FMUL_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmul.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,m",  MATCH_FMUL_B, MASK_FMUL_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FDIV_B | MASK_RM, MASK_FDIV_B | MASK_RM, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fdiv.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T,m",  MATCH_FDIV_B, MASK_FDIV_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsqrt.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S",  MATCH_FSQRT_B | MASK_RM, MASK_FSQRT_B | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fsqrt.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S,m",  MATCH_FSQRT_B, MASK_FSQRT_B, match_opcode,  WR_xd|RD_xs1 },
{"fsgnj.b",   0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FSGNJ_B, MASK_FSGNJ_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjn.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FSGNJN_B, MASK_FSGNJN_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fsgnjx.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FSGNJX_B, MASK_FSGNJX_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmin.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FMIN_B, MASK_FMIN_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"fmax.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,S,T",  MATCH_FMAX_B, MASK_FMAX_B, match_opcode,   WR_xd|RD_xs1|RD_xs2 },
{"feq.b",     0, INSN_CLASS_XPULP_FQUARTER, "d,S,T",    MATCH_FEQ_B, MASK_FEQ_B, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"flt.b",     0, INSN_CLASS_XPULP_FQUARTER, "d,S,T",    MATCH_FLT_B, MASK_FLT_B, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fle.b",     0, INSN_CLASS_XPULP_FQUARTER, "d,S,T",    MATCH_FLE_B, MASK_FLE_B, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fcvt.w.b",  0, INSN_CLASS_XPULP_FQUARTER, "d,S",  MATCH_FCVT_W_B | MASK_RM, MASK_FCVT_W_B | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.w.b",  0, INSN_CLASS_XPULP_FQUARTER, "d,S,m",  MATCH_FCVT_W_B, MASK_FCVT_W_B, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.b", 0, INSN_CLASS_XPULP_FQUARTER, "d,S",  MATCH_FCVT_WU_B | MASK_RM, MASK_FCVT_WU_B | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.wu.b", 0, INSN_CLASS_XPULP_FQUARTER, "d,S,m",  MATCH_FCVT_WU_B, MASK_FCVT_WU_B, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.b.w",  0, INSN_CLASS_XPULP_FQUARTER, "D,s",  MATCH_FCVT_B_W | MASK_RM, MASK_FCVT_B_W | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.w",  0, INSN_CLASS_XPULP_FQUARTER, "D,s,m",  MATCH_FCVT_B_W, MASK_FCVT_B_W, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.wu", 0, INSN_CLASS_XPULP_FQUARTER, "D,s",  MATCH_FCVT_B_WU | MASK_RM, MASK_FCVT_B_W | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.wu", 0, INSN_CLASS_XPULP_FQUARTER, "D,s,m",  MATCH_FCVT_B_WU, MASK_FCVT_B_WU, match_opcode,   WR_xd|RD_xs1 },
{"fmv.x.b",   0, INSN_CLASS_XPULP_FQUARTER, "d,S",  MATCH_FMV_X_B, MASK_FMV_X_B, match_opcode,  WR_xd|RD_xs1 },
{"fclass.b",  0, INSN_CLASS_XPULP_FQUARTER, "D,S",  MATCH_FCLASS_B, MASK_FCLASS_B, match_opcode,   WR_xd|RD_xs1 },
{"fmv.b.x",   0, INSN_CLASS_XPULP_FQUARTER, "D,s",  MATCH_FMV_B_X, MASK_FMV_B_X, match_opcode,  WR_xd|RD_xs1 },
//pseudos
{"fgt.b",     0, INSN_CLASS_XPULP_FQUARTER, "d,T,S",    MATCH_FLT_B, MASK_FLT_B, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fge.b",     0, INSN_CLASS_XPULP_FQUARTER, "d,T,S",    MATCH_FLE_B, MASK_FLE_B, match_opcode,  WR_xd|RD_xs1|RD_xs2 },
{"fmv.b",     0, INSN_CLASS_XPULP_FQUARTER, "D,U",  MATCH_FSGNJ_B, MASK_FSGNJ_B, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fabs.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,U",  MATCH_FSGNJX_B, MASK_FSGNJX_B, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"fneg.b",    0, INSN_CLASS_XPULP_FQUARTER, "D,U",  MATCH_FSGNJN_B, MASK_FSGNJN_B, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },

/* RV64Xfquarter Quarter-Precision Floating-Point Extension (in addition to RV32Xfquarter) */
{"fcvt.l.b",  64, INSN_CLASS_XPULP_FQUARTER, "D,S",  MATCH_FCVT_L_B | MASK_RM, MASK_FCVT_L_B | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.l.b",  64, INSN_CLASS_XPULP_FQUARTER, "D,S,m",  MATCH_FCVT_L_B, MASK_FCVT_L_B, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.b", 64, INSN_CLASS_XPULP_FQUARTER, "D,S",  MATCH_FCVT_LU_B | MASK_RM, MASK_FCVT_LU_B | MASK_RM, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.lu.b", 64, INSN_CLASS_XPULP_FQUARTER, "D,S,m",  MATCH_FCVT_LU_B, MASK_FCVT_LU_B, match_opcode,  WR_xd|RD_xs1 },
{"fcvt.b.l",  64, INSN_CLASS_XPULP_FQUARTER, "D,S",  MATCH_FCVT_B_L | MASK_RM, MASK_FCVT_B_L | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.l",  64, INSN_CLASS_XPULP_FQUARTER, "D,S,m",  MATCH_FCVT_B_L, MASK_FCVT_B_L, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.lu", 64, INSN_CLASS_XPULP_FQUARTER, "D,S",  MATCH_FCVT_B_LU | MASK_RM, MASK_FCVT_B_L | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.lu", 64, INSN_CLASS_XPULP_FQUARTER, "D,S,m",  MATCH_FCVT_B_LU, MASK_FCVT_B_LU, match_opcode,   WR_xd|RD_xs1 },

/* XfquarterwithF - quarter-precision conversions with F extension */
{"fcvt.s.b",  0,  INSN_CLASS_XPULP_FQUARTERWITHF, "D,S",  MATCH_FCVT_S_B, MASK_FCVT_S_B, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.b.s",  0,  INSN_CLASS_XPULP_FQUARTERWITHF, "D,S",  MATCH_FCVT_B_S | MASK_RM, MASK_FCVT_B_S | MASK_RM, match_opcode,   WR_xd|RD_fs1 },
{"fcvt.b.s",  0,  INSN_CLASS_XPULP_FQUARTERWITHF, "D,S,m",  MATCH_FCVT_B_S, MASK_FCVT_B_S, match_opcode,   WR_xd|RD_fs1 },

/* XfquarterwithD - quarter-precision conversions with D extension (in addition to XfquarterwithF) */
{"fcvt.d.b",  0,  INSN_CLASS_XPULP_FQUARTERWITHD, "D,S",  MATCH_FCVT_D_B, MASK_FCVT_D_B, match_opcode,   WR_fd|RD_xs1 },
{"fcvt.b.d",  0,  INSN_CLASS_XPULP_FQUARTERWITHD, "D,S",  MATCH_FCVT_B_D | MASK_RM, MASK_FCVT_B_D | MASK_RM, match_opcode,   WR_xd|RD_fs1 },
{"fcvt.b.d",  0,  INSN_CLASS_XPULP_FQUARTERWITHD, "D,S,m",  MATCH_FCVT_B_D, MASK_FCVT_B_D, match_opcode,   WR_xd|RD_fs1 },

/* Xfquarterwithhalf - quarter-precision conversions with Xfhalf extension */
{"fcvt.h.b",  0,  INSN_CLASS_XPULP_FQUARTERWITHHALF, "D,S",  MATCH_FCVT_H_B, MASK_FCVT_H_B, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.h",  0,  INSN_CLASS_XPULP_FQUARTERWITHHALF, "D,S",  MATCH_FCVT_B_H | MASK_RM, MASK_FCVT_B_H | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.h",  0,  INSN_CLASS_XPULP_FQUARTERWITHHALF, "D,S,m",  MATCH_FCVT_B_H, MASK_FCVT_B_H, match_opcode,   WR_xd|RD_xs1 },

/* Xfquarterwithalthalf - quarter-precision conversions with Xfalthalf extension */
{"fcvt.ah.b", 0,  INSN_CLASS_XPULP_FQUARTERWITHALTHALF, "D,S",  MATCH_FCVT_AH_B, MASK_FCVT_AH_B, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.ah", 0,  INSN_CLASS_XPULP_FQUARTERWITHALTHALF,  "D,S",  MATCH_FCVT_B_AH | MASK_RM, MASK_FCVT_B_AH | MASK_RM, match_opcode,   WR_xd|RD_xs1 },
{"fcvt.b.ah", 0,  INSN_CLASS_XPULP_FQUARTERWITHALTHALF,  "D,S,m",  MATCH_FCVT_B_AH, MASK_FCVT_B_AH, match_opcode,   WR_xd|RD_xs1 },

/* Xfvecsingle - vectorial single-precision floats - requires FLEN >= 64 */
{"vfadd.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFADD_S, MASK_VFADD_S, match_opcode, 0},
{"vfadd.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFADD_R_S, MASK_VFADD_R_S, match_opcode, 0},
{"vfsub.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSUB_S, MASK_VFSUB_S, match_opcode, 0},
{"vfsub.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSUB_R_S, MASK_VFSUB_R_S, match_opcode, 0},
{"vfmul.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMUL_S, MASK_VFMUL_S, match_opcode, 0},
{"vfmul.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMUL_R_S, MASK_VFMUL_R_S, match_opcode, 0},
{"vfdiv.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFDIV_S, MASK_VFDIV_S, match_opcode, 0},
{"vfdiv.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFDIV_R_S, MASK_VFDIV_R_S, match_opcode, 0},
{"vfmin.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMIN_S, MASK_VFMIN_S, match_opcode, 0},
{"vfmin.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMIN_R_S, MASK_VFMIN_R_S, match_opcode, 0},
{"vfmax.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMAX_S, MASK_VFMAX_S, match_opcode, 0},
{"vfmax.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMAX_R_S, MASK_VFMAX_R_S, match_opcode, 0},
{"vfsqrt.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "D,S",    MATCH_VFSQRT_S, MASK_VFSQRT_S, match_opcode, 0},
{"vfmac.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMAC_S, MASK_VFMAC_S, match_opcode, 0},
{"vfmac.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMAC_R_S, MASK_VFMAC_R_S, match_opcode, 0},
{"vfmre.s",     0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMRE_S, MASK_VFMRE_S, match_opcode, 0},
{"vfmre.r.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFMRE_R_S, MASK_VFMRE_R_S, match_opcode, 0},
{"vfclass.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S",  MATCH_VFCLASS_S, MASK_VFCLASS_S, match_opcode, 0},
{"vfsgnj.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSGNJ_S, MASK_VFSGNJ_S, match_opcode, 0},
{"vfsgnj.r.s",  0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSGNJ_R_S, MASK_VFSGNJ_R_S, match_opcode, 0},
{"vfsgnjn.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSGNJN_S, MASK_VFSGNJN_S, match_opcode, 0},
{"vfsgnjn.r.s", 0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSGNJN_R_S, MASK_VFSGNJN_R_S, match_opcode, 0},
{"vfsgnjx.s",   0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSGNJX_S, MASK_VFSGNJX_S, match_opcode, 0},
{"vfsgnjx.r.s", 0, INSN_CLASS_XPULP_FVECSINGLE, "D,S,T",        MATCH_VFSGNJX_R_S, MASK_VFSGNJX_R_S, match_opcode, 0},
{"vfeq.s",      0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFEQ_S, MASK_VFEQ_S, match_opcode, 0},
{"vfeq.r.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFEQ_R_S, MASK_VFEQ_R_S, match_opcode, 0},
{"vfne.s",      0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFNE_S, MASK_VFNE_S, match_opcode, 0},
{"vfne.r.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFNE_R_S, MASK_VFNE_R_S, match_opcode, 0},
{"vflt.s",      0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFLT_S, MASK_VFLT_S, match_opcode, 0},
{"vflt.r.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFLT_R_S, MASK_VFLT_R_S, match_opcode, 0},
{"vfge.s",      0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFGE_S, MASK_VFGE_S, match_opcode, 0},
{"vfge.r.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFGE_R_S, MASK_VFGE_R_S, match_opcode, 0},
{"vfle.s",      0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFLE_S, MASK_VFLE_S, match_opcode, 0},
{"vfle.r.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFLE_R_S, MASK_VFLE_R_S, match_opcode, 0},
{"vfgt.s",      0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFGT_S, MASK_VFGT_S, match_opcode, 0},
{"vfgt.r.s",    0, INSN_CLASS_XPULP_FVECSINGLE, "d,S,T",        MATCH_VFGT_R_S, MASK_VFGT_R_S, match_opcode, 0},
// pseudos
{"vfabs.s",    0,  INSN_CLASS_XPULP_FVECSINGLE, "D,U",  MATCH_VFSGNJX_S, MASK_VFSGNJX_S, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"vfneg.s",    0,  INSN_CLASS_XPULP_FVECSINGLE, "D,U",  MATCH_VFSGNJN_S, MASK_VFSGNJN_S, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
/* XfvecsinglenothirtytwoD: only unless RV32D */
{"vfmv.x.s",    0, INSN_CLASS_XPULP_FVECSINGLENOTTHIRTYTWOD, "d,S",     MATCH_VFMV_X_S, MASK_VFMV_X_S, match_opcode, 0},
{"vfmv.s.x",    0, INSN_CLASS_XPULP_FVECSINGLENOTTHIRTYTWOD, "S,d",     MATCH_VFMV_S_X, MASK_VFMV_S_X, match_opcode, 0},
{"vfcvt.x.s",   0, INSN_CLASS_XPULP_FVECSINGLENOTTHIRTYTWOD, "d,S",     MATCH_VFCVT_X_S, MASK_VFCVT_X_S, match_opcode, 0},
{"vfcvt.xu.s",  0, INSN_CLASS_XPULP_FVECSINGLENOTTHIRTYTWOD, "d,S",     MATCH_VFCVT_XU_S, MASK_VFCVT_XU_S, match_opcode, 0},
{"vfcvt.s.x",   0, INSN_CLASS_XPULP_FVECSINGLENOTTHIRTYTWOD, "S,d",     MATCH_VFCVT_S_X, MASK_VFCVT_S_X, match_opcode, 0},
{"vfcvt.s.xu",  0, INSN_CLASS_XPULP_FVECSINGLENOTTHIRTYTWOD, "S,d",     MATCH_VFCVT_S_XU, MASK_VFCVT_S_XU, match_opcode, 0},
/* XfvecsinglewithF - vectorial single-precision conversions with F extension, a-d legality depends on FLEN */
{"vfcpka.s.s",    0, INSN_CLASS_XPULP_FVECSINGLEWITHF, "D,S,T",         MATCH_VFCPKA_S_S, MASK_VFCPKA_S_S, match_opcode, 0},
{"vfcpkb.s.s",    0, INSN_CLASS_XPULP_FVECSINGLEWITHF, "D,S,T", MATCH_VFCPKB_S_S, MASK_VFCPKB_S_S, match_opcode, 0},
{"vfcpkc.s.s",    0, INSN_CLASS_XPULP_FVECSINGLEWITHF, "D,S,T", MATCH_VFCPKC_S_S, MASK_VFCPKC_S_S, match_opcode, 0},
{"vfcpkd.s.s",    0, INSN_CLASS_XPULP_FVECSINGLEWITHF, "D,S,T", MATCH_VFCPKD_S_S, MASK_VFCPKD_S_S, match_opcode, 0},
/* XfvecsinglewithD - vectorial single-precision conversions with D extension, a-d legality depends on FLEN (in addition to the above) */
{"vfcpka.s.d",    0, INSN_CLASS_XPULP_FVECSINGLEWITHD, "D,S,T",         MATCH_VFCPKA_S_D, MASK_VFCPKA_S_D, match_opcode, 0},
{"vfcpkb.s.d",    0, INSN_CLASS_XPULP_FVECSINGLEWITHD, "D,S,T", MATCH_VFCPKB_S_D, MASK_VFCPKB_S_D, match_opcode, 0},
{"vfcpkc.s.d",    0, INSN_CLASS_XPULP_FVECSINGLEWITHD, "D,S,T", MATCH_VFCPKC_S_D, MASK_VFCPKC_S_D, match_opcode, 0},
{"vfcpkd.s.d",    0, INSN_CLASS_XPULP_FVECSINGLEWITHD, "D,S,T", MATCH_VFCPKD_S_D, MASK_VFCPKD_S_D, match_opcode, 0},

/* Xfvechalf - vectorial half-precision floats - requires FLEN >= 32 */
{"vfadd.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFADD_H, MASK_VFADD_H, match_opcode, 0},
{"vfadd.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFADD_R_H, MASK_VFADD_R_H, match_opcode, 0},
{"vfsub.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSUB_H, MASK_VFSUB_H, match_opcode, 0},
{"vfsub.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSUB_R_H, MASK_VFSUB_R_H, match_opcode, 0},
{"vfmul.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMUL_H, MASK_VFMUL_H, match_opcode, 0},
{"vfmul.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMUL_R_H, MASK_VFMUL_R_H, match_opcode, 0},
{"vfdiv.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFDIV_H, MASK_VFDIV_H, match_opcode, 0},
{"vfdiv.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFDIV_R_H, MASK_VFDIV_R_H, match_opcode, 0},
{"vfmin.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMIN_H, MASK_VFMIN_H, match_opcode, 0},
{"vfmin.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMIN_R_H, MASK_VFMIN_R_H, match_opcode, 0},
{"vfmax.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMAX_H, MASK_VFMAX_H, match_opcode, 0},
{"vfmax.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMAX_R_H, MASK_VFMAX_R_H, match_opcode, 0},
{"vfsqrt.h",    0, INSN_CLASS_XPULP_FVECHALF, "D,S",    MATCH_VFSQRT_H, MASK_VFSQRT_H, match_opcode, 0},
{"vfmac.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMAC_H, MASK_VFMAC_H, match_opcode, 0},
{"vfmac.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMAC_R_H, MASK_VFMAC_R_H, match_opcode, 0},
{"vfmre.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMRE_H, MASK_VFMRE_H, match_opcode, 0},
{"vfmre.r.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFMRE_R_H, MASK_VFMRE_R_H, match_opcode, 0},
{"vfclass.h",   0, INSN_CLASS_XPULP_FVECHALF, "d,S",    MATCH_VFCLASS_H, MASK_VFCLASS_H, match_opcode, 0},
{"vfsgnj.h",    0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSGNJ_H, MASK_VFSGNJ_H, match_opcode, 0},
{"vfsgnj.r.h",  0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSGNJ_R_H, MASK_VFSGNJ_R_H, match_opcode, 0},
{"vfsgnjn.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSGNJN_H, MASK_VFSGNJN_H, match_opcode, 0},
{"vfsgnjn.r.h", 0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSGNJN_R_H, MASK_VFSGNJN_R_H, match_opcode, 0},
{"vfsgnjx.h",   0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSGNJX_H, MASK_VFSGNJX_H, match_opcode, 0},
{"vfsgnjx.r.h", 0, INSN_CLASS_XPULP_FVECHALF, "D,S,T",  MATCH_VFSGNJX_R_H, MASK_VFSGNJX_R_H, match_opcode, 0},
{"vfeq.h",      0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFEQ_H, MASK_VFEQ_H, match_opcode, 0},
{"vfeq.r.h",    0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFEQ_R_H, MASK_VFEQ_R_H, match_opcode, 0},
{"vfne.h",      0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFNE_H, MASK_VFNE_H, match_opcode, 0},
{"vfne.r.h",    0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFNE_R_H, MASK_VFNE_R_H, match_opcode, 0},
{"vflt.h",      0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFLT_H, MASK_VFLT_H, match_opcode, 0},
{"vflt.r.h",    0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFLT_R_H, MASK_VFLT_R_H, match_opcode, 0},
{"vfge.h",      0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFGE_H, MASK_VFGE_H, match_opcode, 0},
{"vfge.r.h",    0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFGE_R_H, MASK_VFGE_R_H, match_opcode, 0},
{"vfle.h",      0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFLE_H, MASK_VFLE_H, match_opcode, 0},
{"vfle.r.h",    0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFLE_R_H, MASK_VFLE_R_H, match_opcode, 0},
{"vfgt.h",      0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFGT_H, MASK_VFGT_H, match_opcode, 0},
{"vfgt.r.h",    0, INSN_CLASS_XPULP_FVECHALF, "d,S,T",  MATCH_VFGT_R_H, MASK_VFGT_R_H, match_opcode, 0},
// pseudos
{"vfabs.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,U",  MATCH_VFSGNJX_H, MASK_VFSGNJX_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"vfneg.h",     0, INSN_CLASS_XPULP_FVECHALF, "D,U",  MATCH_VFSGNJN_H, MASK_VFSGNJN_H, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
/* XfvechalfnothirtytwoD: only unless RV32D */
{"vfmv.x.h",    0, INSN_CLASS_XPULP_FVECHALFNOTTHIRTYTWOD, "d,S",       MATCH_VFMV_X_H, MASK_VFMV_X_H, match_opcode, 0},
{"vfmv.h.x",    0, INSN_CLASS_XPULP_FVECHALFNOTTHIRTYTWOD, "D,s",       MATCH_VFMV_H_X, MASK_VFMV_H_X, match_opcode, 0},
{"vfcvt.x.h",   0, INSN_CLASS_XPULP_FVECHALFNOTTHIRTYTWOD, "d,S",       MATCH_VFCVT_X_H, MASK_VFCVT_X_H, match_opcode, 0},
{"vfcvt.xu.h",  0, INSN_CLASS_XPULP_FVECHALFNOTTHIRTYTWOD, "d,S",       MATCH_VFCVT_XU_H, MASK_VFCVT_XU_H, match_opcode, 0},
{"vfcvt.h.x",   0, INSN_CLASS_XPULP_FVECHALFNOTTHIRTYTWOD, "D,s",       MATCH_VFCVT_H_X, MASK_VFCVT_H_X, match_opcode, 0},
{"vfcvt.h.xu",  0, INSN_CLASS_XPULP_FVECHALFNOTTHIRTYTWOD, "D,s",       MATCH_VFCVT_H_XU, MASK_VFCVT_H_XU, match_opcode, 0},
/* XfvechalfwithF - vectorial half-precision conversions with F extension, a-d legality depends on FLEN */
{"vfcpka.h.s",  0, INSN_CLASS_XPULP_FVECHALFWITHF, "D,S,T",     MATCH_VFCPKA_H_S, MASK_VFCPKA_H_S, match_opcode, 0},
{"vfcpkb.h.s",  0, INSN_CLASS_XPULP_FVECHALFWITHF, "D,S,T",     MATCH_VFCPKB_H_S, MASK_VFCPKB_H_S, match_opcode, 0},
{"vfcpkc.h.s",  0, INSN_CLASS_XPULP_FVECHALFWITHF, "D,S,T",  MATCH_VFCPKC_H_S, MASK_VFCPKC_H_S, match_opcode, 0},
{"vfcpkd.h.s",  0, INSN_CLASS_XPULP_FVECHALFWITHF, "D,S,T",  MATCH_VFCPKD_H_S, MASK_VFCPKD_H_S, match_opcode, 0},
/* XfvechalfwithD - vectorial half-precision conversions with D extension, a-d legality depends on FLEN (in addition to the above) */
{"vfcpka.h.d",  0, INSN_CLASS_XPULP_FVECHALFWITHD, "D,S,T",     MATCH_VFCPKA_H_D, MASK_VFCPKA_H_D, match_opcode, 0},
{"vfcpkb.h.d",  0, INSN_CLASS_XPULP_FVECHALFWITHD, "D,S,T",     MATCH_VFCPKB_H_D, MASK_VFCPKB_H_D, match_opcode, 0},
{"vfcpkc.h.d",  0, INSN_CLASS_XPULP_FVECHALFWITHD, "D,S,T",  MATCH_VFCPKC_H_D, MASK_VFCPKC_H_D, match_opcode, 0},
{"vfcpkd.h.d",  0, INSN_CLASS_XPULP_FVECHALFWITHD, "D,S,T",  MATCH_VFCPKD_H_D, MASK_VFCPKD_H_D, match_opcode, 0},
/* Xfvecalthalfwithsingle - vectorial half-precision conversions with Xfvecsingle extension */
{"vfcvt.s.h",   0, INSN_CLASS_XPULP_FVECHALFWITHSINGLE, "D,S",  MATCH_VFCVT_S_H, MASK_VFCVT_S_H, match_opcode, 0},
{"vfcvtu.s.h",  0, INSN_CLASS_XPULP_FVECHALFWITHSINGLE, "D,S",    MATCH_VFCVTU_S_H, MASK_VFCVTU_S_H, match_opcode, 0},
{"vfcvt.h.s",   0, INSN_CLASS_XPULP_FVECHALFWITHSINGLE, "D,S",  MATCH_VFCVT_H_S, MASK_VFCVT_H_S, match_opcode, 0},
{"vfcvtu.h.s",  0, INSN_CLASS_XPULP_FVECHALFWITHSINGLE, "D,S",    MATCH_VFCVTU_H_S, MASK_VFCVTU_H_S, match_opcode, 0},

/* Xfvecalthalf - vectorial alternate half-precision floats - requires FLEN >= 32 */
{"vfadd.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFADD_AH, MASK_VFADD_AH, match_opcode, 0},
{"vfadd.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFADD_R_AH, MASK_VFADD_R_AH, match_opcode, 0},
{"vfsub.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSUB_AH, MASK_VFSUB_AH, match_opcode, 0},
{"vfsub.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSUB_R_AH, MASK_VFSUB_R_AH, match_opcode, 0},
{"vfmul.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMUL_AH, MASK_VFMUL_AH, match_opcode, 0},
{"vfmul.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMUL_R_AH, MASK_VFMUL_R_AH, match_opcode, 0},
{"vfdiv.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFDIV_AH, MASK_VFDIV_AH, match_opcode, 0},
{"vfdiv.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFDIV_R_AH, MASK_VFDIV_R_AH, match_opcode, 0},
{"vfmin.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMIN_AH, MASK_VFMIN_AH, match_opcode, 0},
{"vfmin.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMIN_R_AH, MASK_VFMIN_R_AH, match_opcode, 0},
{"vfmax.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMAX_AH, MASK_VFMAX_AH, match_opcode, 0},
{"vfmax.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMAX_R_AH, MASK_VFMAX_R_AH, match_opcode, 0},
{"vfsqrt.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "D,S",         MATCH_VFSQRT_AH, MASK_VFSQRT_AH, match_opcode, 0},
{"vfmac.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMAC_AH, MASK_VFMAC_AH, match_opcode, 0},
{"vfmac.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMAC_R_AH, MASK_VFMAC_R_AH, match_opcode, 0},
{"vfmre.ah",    0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMRE_AH, MASK_VFMRE_AH, match_opcode, 0},
{"vfmre.r.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFMRE_R_AH, MASK_VFMRE_R_AH, match_opcode, 0},
{"vfclass.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S",         MATCH_VFCLASS_AH, MASK_VFCLASS_AH, match_opcode, 0},
{"vfsgnj.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSGNJ_AH, MASK_VFSGNJ_AH, match_opcode, 0},
{"vfsgnj.r.ah", 0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSGNJ_R_AH, MASK_VFSGNJ_R_AH, match_opcode, 0},
{"vfsgnjn.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSGNJN_AH, MASK_VFSGNJN_AH, match_opcode, 0},
{"vfsgnjn.r.ah",0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSGNJN_R_AH, MASK_VFSGNJN_R_AH, match_opcode, 0},
{"vfsgnjx.ah",  0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSGNJX_AH, MASK_VFSGNJX_AH, match_opcode, 0},
{"vfsgnjx.r.ah",0, INSN_CLASS_XPULP_FVECALTHALF, "D,S,T",       MATCH_VFSGNJX_R_AH, MASK_VFSGNJX_R_AH, match_opcode, 0},
{"vfeq.ah",     0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFEQ_AH, MASK_VFEQ_AH, match_opcode, 0},
{"vfeq.r.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFEQ_R_AH, MASK_VFEQ_R_AH, match_opcode, 0},
{"vfne.ah",     0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFNE_AH, MASK_VFNE_AH, match_opcode, 0},
{"vfne.r.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFNE_R_AH, MASK_VFNE_R_AH, match_opcode, 0},
{"vflt.ah",     0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFLT_AH, MASK_VFLT_AH, match_opcode, 0},
{"vflt.r.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFLT_R_AH, MASK_VFLT_R_AH, match_opcode, 0},
{"vfge.ah",     0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFGE_AH, MASK_VFGE_AH, match_opcode, 0},
{"vfge.r.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFGE_R_AH, MASK_VFGE_R_AH, match_opcode, 0},
{"vfle.ah",     0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFLE_AH, MASK_VFLE_AH, match_opcode, 0},
{"vfle.r.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFLE_R_AH, MASK_VFLE_R_AH, match_opcode, 0},
{"vfgt.ah",     0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFGT_AH, MASK_VFGT_AH, match_opcode, 0},
{"vfgt.r.ah",   0, INSN_CLASS_XPULP_FVECALTHALF, "d,S,T",       MATCH_VFGT_R_AH, MASK_VFGT_R_AH, match_opcode, 0},
// pseudos
{"vfabs.ah",    0, INSN_CLASS_XPULP_FVECALTHALF,  "D,U",  MATCH_VFSGNJX_AH, MASK_VFSGNJX_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"vfneg.ah",    0, INSN_CLASS_XPULP_FVECALTHALF,  "D,U",  MATCH_VFSGNJN_AH, MASK_VFSGNJN_AH, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
/* XfvecalthalfnothirtytwoD: only unless RV32D */
{"vfmv.x.ah",   0, INSN_CLASS_XPULP_FVECALTHALFNOTTHIRTYTWOD, "d,S",    MATCH_VFMV_X_AH, MASK_VFMV_X_AH, match_opcode, 0},
{"vfmv.ah.x",   0, INSN_CLASS_XPULP_FVECALTHALFNOTTHIRTYTWOD, "D,s",    MATCH_VFMV_AH_X, MASK_VFMV_AH_X, match_opcode, 0},
{"vfcvt.x.ah",  0, INSN_CLASS_XPULP_FVECALTHALFNOTTHIRTYTWOD, "d,S",    MATCH_VFCVT_X_AH, MASK_VFCVT_X_AH, match_opcode, 0},
{"vfcvt.xu.ah", 0, INSN_CLASS_XPULP_FVECALTHALFNOTTHIRTYTWOD, "d,S",    MATCH_VFCVT_XU_AH, MASK_VFCVT_XU_AH, match_opcode, 0},
{"vfcvt.ah.x",  0, INSN_CLASS_XPULP_FVECALTHALFNOTTHIRTYTWOD, "D,s",    MATCH_VFCVT_AH_X, MASK_VFCVT_AH_X, match_opcode, 0},
{"vfcvt.ah.xu", 0, INSN_CLASS_XPULP_FVECALTHALFNOTTHIRTYTWOD, "D,s",    MATCH_VFCVT_AH_XU, MASK_VFCVT_AH_XU, match_opcode, 0},
/* XfvecalthalfwithF - vectorial alternate half-precision conversions with F extension, a-d legality depends on FLEN */
{"vfcpka.ah.s", 0, INSN_CLASS_XPULP_FVECALTHALFWITHF, "D,S,T",  MATCH_VFCPKA_AH_S, MASK_VFCPKA_AH_S, match_opcode, 0},
{"vfcpkb.ah.s", 0, INSN_CLASS_XPULP_FVECALTHALFWITHF, "D,S,T",  MATCH_VFCPKB_AH_S, MASK_VFCPKB_AH_S, match_opcode, 0},
{"vfcpkc.ah.s", 0, INSN_CLASS_XPULP_FVECALTHALFWITHF, "D,S,T", MATCH_VFCPKC_AH_S, MASK_VFCPKC_AH_S, match_opcode, 0},
{"vfcpkd.ah.s", 0, INSN_CLASS_XPULP_FVECALTHALFWITHF, "D,S,T", MATCH_VFCPKD_AH_S, MASK_VFCPKD_AH_S, match_opcode, 0},
/* XfvecalthalfwithD - vectorial alternate half-precision conversions with D extension, a-d legality depends on FLEN (in addition to the above) */
{"vfcpka.ah.d", 0, INSN_CLASS_XPULP_FVECALTHALFWITHD, "D,S,T",  MATCH_VFCPKA_AH_D, MASK_VFCPKA_AH_D, match_opcode, 0},
{"vfcpkb.ah.d", 0, INSN_CLASS_XPULP_FVECALTHALFWITHD, "D,S,T",  MATCH_VFCPKB_AH_D, MASK_VFCPKB_AH_D, match_opcode, 0},
{"vfcpkc.ah.d", 0, INSN_CLASS_XPULP_FVECALTHALFWITHD, "D,S,T", MATCH_VFCPKC_AH_D, MASK_VFCPKC_AH_D, match_opcode, 0},
{"vfcpkd.ah.d", 0, INSN_CLASS_XPULP_FVECALTHALFWITHD, "D,S,T", MATCH_VFCPKD_AH_D, MASK_VFCPKD_AH_D, match_opcode, 0},
/* Xfvecalthalfwithsingle - vectorial alternate half-precision conversions with Xfvecsingle extension */
{"vfcvt.s.ah",  0, INSN_CLASS_XPULP_FVECALTHALFWITHSINGLE, "D,S",       MATCH_VFCVT_S_AH, MASK_VFCVT_S_AH, match_opcode, 0},
{"vfcvtu.s.ah", 0, INSN_CLASS_XPULP_FVECALTHALFWITHSINGLE, "D,S", MATCH_VFCVTU_S_AH, MASK_VFCVTU_S_AH, match_opcode, 0},
{"vfcvt.ah.s",  0, INSN_CLASS_XPULP_FVECALTHALFWITHSINGLE, "D,S",       MATCH_VFCVT_AH_S, MASK_VFCVT_AH_S, match_opcode, 0},
{"vfcvtu.ah.s", 0, INSN_CLASS_XPULP_FVECALTHALFWITHSINGLE, "D,S", MATCH_VFCVTU_AH_S, MASK_VFCVTU_AH_S, match_opcode, 0},
/* Xfvecalthalfwithhalf - vectorial alternate half-precision conversions with Xfvechalf extension */
{"vfcvt.h.ah",  0, INSN_CLASS_XPULP_FVECALTHALFWITHHALF, "D,S",         MATCH_VFCVT_H_AH, MASK_VFCVT_H_AH, match_opcode, 0},
{"vfcvtu.h.ah", 0, INSN_CLASS_XPULP_FVECALTHALFWITHHALF, "D,S", MATCH_VFCVTU_H_AH, MASK_VFCVTU_H_AH, match_opcode, 0},
{"vfcvt.ah.h",  0, INSN_CLASS_XPULP_FVECALTHALFWITHHALF, "D,S",         MATCH_VFCVT_AH_H, MASK_VFCVT_AH_H, match_opcode, 0},
{"vfcvtu.ah.h", 0, INSN_CLASS_XPULP_FVECALTHALFWITHHALF, "D,S", MATCH_VFCVTU_AH_H, MASK_VFCVTU_AH_H, match_opcode, 0},

/* Xfvecquarter - vectorial quarter-precision floats - requires FLEN >= 16 */
{"vfadd.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFADD_B, MASK_VFADD_B, match_opcode, 0},
{"vfadd.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFADD_R_B, MASK_VFADD_R_B, match_opcode, 0},
{"vfsub.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSUB_B, MASK_VFSUB_B, match_opcode, 0},
{"vfsub.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSUB_R_B, MASK_VFSUB_R_B, match_opcode, 0},
{"vfmul.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMUL_B, MASK_VFMUL_B, match_opcode, 0},
{"vfmul.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMUL_R_B, MASK_VFMUL_R_B, match_opcode, 0},
{"vfdiv.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFDIV_B, MASK_VFDIV_B, match_opcode, 0},
{"vfdiv.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFDIV_R_B, MASK_VFDIV_R_B, match_opcode, 0},
{"vfmin.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMIN_B, MASK_VFMIN_B, match_opcode, 0},
{"vfmin.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMIN_R_B, MASK_VFMIN_R_B, match_opcode, 0},
{"vfmax.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMAX_B, MASK_VFMAX_B, match_opcode, 0},
{"vfmax.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMAX_R_B, MASK_VFMAX_R_B, match_opcode, 0},
{"vfsqrt.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "D,S",         MATCH_VFSQRT_B, MASK_VFSQRT_B, match_opcode, 0},
{"vfmac.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMAC_B, MASK_VFMAC_B, match_opcode, 0},
{"vfmac.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMAC_R_B, MASK_VFMAC_R_B, match_opcode, 0},
{"vfmre.b",     0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMRE_B, MASK_VFMRE_B, match_opcode, 0},
{"vfmre.r.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFMRE_R_B, MASK_VFMRE_R_B, match_opcode, 0},
{"vfclass.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S",         MATCH_VFCLASS_B, MASK_VFCLASS_B, match_opcode, 0},
{"vfsgnj.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSGNJ_B, MASK_VFSGNJ_B, match_opcode, 0},
{"vfsgnj.r.b",  0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSGNJ_R_B, MASK_VFSGNJ_R_B, match_opcode, 0},
{"vfsgnjn.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSGNJN_B, MASK_VFSGNJN_B, match_opcode, 0},
{"vfsgnjn.r.b", 0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSGNJN_R_B, MASK_VFSGNJN_R_B, match_opcode, 0},
{"vfsgnjx.b",   0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSGNJX_B, MASK_VFSGNJX_B, match_opcode, 0},
{"vfsgnjx.r.b", 0, INSN_CLASS_XPULP_FVECQUARTER, "D,S,T",       MATCH_VFSGNJX_R_B, MASK_VFSGNJX_R_B, match_opcode, 0},
{"vfeq.b",      0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFEQ_B, MASK_VFEQ_B, match_opcode, 0},
{"vfeq.r.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFEQ_R_B, MASK_VFEQ_R_B, match_opcode, 0},
{"vfne.b",      0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFNE_B, MASK_VFNE_B, match_opcode, 0},
{"vfne.r.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFNE_R_B, MASK_VFNE_R_B, match_opcode, 0},
{"vflt.b",      0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFLT_B, MASK_VFLT_B, match_opcode, 0},
{"vflt.r.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFLT_R_B, MASK_VFLT_R_B, match_opcode, 0},
{"vfge.b",      0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFGE_B, MASK_VFGE_B, match_opcode, 0},
{"vfge.r.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFGE_R_B, MASK_VFGE_R_B, match_opcode, 0},
{"vfle.b",      0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFLE_B, MASK_VFLE_B, match_opcode, 0},
{"vfle.r.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFLE_R_B, MASK_VFLE_R_B, match_opcode, 0},
{"vfgt.b",      0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFGT_B, MASK_VFGT_B, match_opcode, 0},
{"vfgt.r.b",    0, INSN_CLASS_XPULP_FVECQUARTER, "d,S,T",       MATCH_VFGT_R_B, MASK_VFGT_R_B, match_opcode, 0},
// pseudos
{"vfabs.b",    0, INSN_CLASS_XPULP_FVECQUARTER,  "D,U",  MATCH_VFSGNJX_B, MASK_VFSGNJX_B, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
{"vfneg.b",    0, INSN_CLASS_XPULP_FVECQUARTER,  "D,U",  MATCH_VFSGNJN_B, MASK_VFSGNJN_B, match_rs1_eq_rs2,   INSN_ALIAS|WR_xd|RD_xs1|RD_xs2 },
/* XfvecquarternothirtytwoD: only unless RV32D */
{"vfmv.x.b",    0, INSN_CLASS_XPULP_FVECQUARTERNOTTHIRTYTWOD, "d,S",    MATCH_VFMV_X_B, MASK_VFMV_X_B, match_opcode, 0},
{"vfmv.b.x",    0, INSN_CLASS_XPULP_FVECQUARTERNOTTHIRTYTWOD, "D,s",    MATCH_VFMV_B_X, MASK_VFMV_B_X, match_opcode, 0},
{"vfcvt.x.b",   0, INSN_CLASS_XPULP_FVECQUARTERNOTTHIRTYTWOD, "d,S",    MATCH_VFCVT_X_B, MASK_VFCVT_X_B, match_opcode, 0},
{"vfcvt.xu.b",  0, INSN_CLASS_XPULP_FVECQUARTERNOTTHIRTYTWOD, "d,S",    MATCH_VFCVT_XU_B, MASK_VFCVT_XU_B, match_opcode, 0},
{"vfcvt.b.x",   0, INSN_CLASS_XPULP_FVECQUARTERNOTTHIRTYTWOD, "D,s",    MATCH_VFCVT_B_X, MASK_VFCVT_B_X, match_opcode, 0},
{"vfcvt.b.xu",  0, INSN_CLASS_XPULP_FVECQUARTERNOTTHIRTYTWOD, "D,s",    MATCH_VFCVT_B_XU, MASK_VFCVT_B_XU, match_opcode, 0},
/* XfvecquarterwithF - vectorial quarter-precision conversions with F extension, a-d legality depends on FLEN */
{"vfcpka.b.s",  0, INSN_CLASS_XPULP_FVECQUARTERWITHF, "D,S,T",  MATCH_VFCPKA_B_S, MASK_VFCPKA_B_S, match_opcode, 0},
{"vfcpkb.b.s",  0, INSN_CLASS_XPULP_FVECQUARTERWITHF, "D,S,T",  MATCH_VFCPKB_B_S, MASK_VFCPKB_B_S, match_opcode, 0},
{"vfcpkc.b.s",  0, INSN_CLASS_XPULP_FVECQUARTERWITHF, "D,S,T",  MATCH_VFCPKC_B_S, MASK_VFCPKC_B_S, match_opcode, 0},
{"vfcpkd.b.s",  0, INSN_CLASS_XPULP_FVECQUARTERWITHF, "D,S,T",  MATCH_VFCPKD_B_S, MASK_VFCPKD_B_S, match_opcode, 0},
/* XfvecquarterwithD - vectorial quarter-precision conversions with D extension, a-d legality depends on FLEN (in addition to above) */
{"vfcpka.b.d",  0, INSN_CLASS_XPULP_FVECQUARTERWITHD, "D,S,T",  MATCH_VFCPKA_B_D, MASK_VFCPKA_B_D, match_opcode, 0},
{"vfcpkb.b.d",  0, INSN_CLASS_XPULP_FVECQUARTERWITHD, "D,S,T",  MATCH_VFCPKB_B_D, MASK_VFCPKB_B_D, match_opcode, 0},
{"vfcpkc.b.d",  0, INSN_CLASS_XPULP_FVECQUARTERWITHD, "D,S,T",  MATCH_VFCPKC_B_D, MASK_VFCPKC_B_D, match_opcode, 0},
{"vfcpkd.b.d",  0, INSN_CLASS_XPULP_FVECQUARTERWITHD, "D,S,T",  MATCH_VFCPKD_B_D, MASK_VFCPKD_B_D, match_opcode, 0},
/* Xfvecquarterwithsingle - vectorial quarter-precision conversions with Xfvecsingle extension */
{"vfcvt.s.b",   0, INSN_CLASS_XPULP_FVECQUARTERWITHSINGLE, "D,S",       MATCH_VFCVT_S_B, MASK_VFCVT_S_B, match_opcode, 0},
{"vfcvtu.s.b",  0, INSN_CLASS_XPULP_FVECQUARTERWITHSINGLE, "D,S", MATCH_VFCVTU_S_B, MASK_VFCVTU_S_B, match_opcode, 0},
{"vfcvt.b.s",   0, INSN_CLASS_XPULP_FVECQUARTERWITHSINGLE, "D,S",       MATCH_VFCVT_B_S, MASK_VFCVT_B_S, match_opcode, 0},
{"vfcvtu.b.s",  0, INSN_CLASS_XPULP_FVECQUARTERWITHSINGLE, "D,S", MATCH_VFCVTU_B_S, MASK_VFCVTU_B_S, match_opcode, 0},
/* Xfvecquarterwithhalf - vectorial quarter-precision conversions with Xfvechalf extension */
{"vfcvt.h.b",   0, INSN_CLASS_XPULP_FVECQUARTERWITHHALF, "D,S",         MATCH_VFCVT_H_B, MASK_VFCVT_H_B, match_opcode, 0},
{"vfcvtu.h.b",  0, INSN_CLASS_XPULP_FVECQUARTERWITHHALF, "D,S", MATCH_VFCVTU_H_B, MASK_VFCVTU_H_B, match_opcode, 0},
{"vfcvt.b.h",   0, INSN_CLASS_XPULP_FVECQUARTERWITHHALF, "D,S",         MATCH_VFCVT_B_H, MASK_VFCVT_B_H, match_opcode, 0},
{"vfcvtu.b.h",  0, INSN_CLASS_XPULP_FVECQUARTERWITHHALF, "D,S", MATCH_VFCVTU_B_H, MASK_VFCVTU_B_H, match_opcode, 0},
/* Xfvecquarterwithalthalf - vectorial quarter-precision conversions with Xfvecalthalf extension */
{"vfcvt.ah.b",  0, INSN_CLASS_XPULP_FVECQUARTERWITHALTHALF, "D,S",      MATCH_VFCVT_AH_B, MASK_VFCVT_AH_B, match_opcode, 0},
{"vfcvtu.ah.b", 0, INSN_CLASS_XPULP_FVECQUARTERWITHALTHALF, "D,S", MATCH_VFCVTU_AH_B, MASK_VFCVTU_AH_B, match_opcode, 0},
{"vfcvt.b.ah",  0, INSN_CLASS_XPULP_FVECQUARTERWITHALTHALF, "D,S",      MATCH_VFCVT_B_AH, MASK_VFCVT_B_AH, match_opcode, 0},
{"vfcvtu.b.ah", 0, INSN_CLASS_XPULP_FVECQUARTERWITHALTHALF, "D,S", MATCH_VFCVTU_B_AH, MASK_VFCVTU_B_AH, match_opcode, 0},

/* Xfauxsingle - single-precision auxiliary operations - requires F extension */
/* NONE */
/* Xfauxvecsingle - vectorial single-precision auxiliary operations - requires Xfvecsingle extension */
{"vfdotp.s",     0, INSN_CLASS_XPULP_FAUXVECSINGLE, "D,S,T",    MATCH_VFDOTP_S, MASK_VFDOTP_S, match_opcode, 0},
{"vfdotp.r.s",   0, INSN_CLASS_XPULP_FAUXVECSINGLE, "D,S,T",    MATCH_VFDOTP_R_S, MASK_VFDOTP_R_S, match_opcode, 0},
{"vfavg.s",      0, INSN_CLASS_XPULP_FAUXVECSINGLE, "D,S,T",    MATCH_VFAVG_S, MASK_VFAVG_S, match_opcode, 0},
{"vfavg.r.s",    0, INSN_CLASS_XPULP_FAUXVECSINGLE, "D,S,T",    MATCH_VFAVG_R_S, MASK_VFAVG_R_S, match_opcode, 0},
/* Xfauxhalf - half-precision auxiliary operations - requires Xfhalf extension */
{"fmulex.s.h",   0, INSN_CLASS_XPULP_FAUXHALF, "D,S,T",         MATCH_FMULEX_S_H | MASK_RM, MASK_FMULEX_S_H | MASK_RM, match_opcode, 0},
{"fmulex.s.h",   0, INSN_CLASS_XPULP_FAUXHALF, "D,S,T,m",       MATCH_FMULEX_S_H, MASK_FMULEX_S_H, match_opcode, 0},
{"fmacex.s.h",   0, INSN_CLASS_XPULP_FAUXHALF, "D,S,T",         MATCH_FMACEX_S_H | MASK_RM, MASK_FMACEX_S_H | MASK_RM, match_opcode, 0},
{"fmacex.s.h",   0, INSN_CLASS_XPULP_FAUXHALF, "D,S,T,m",       MATCH_FMACEX_S_H, MASK_FMACEX_S_H, match_opcode, 0},
/* Xfauxvechalf - vectorial half-precision auxiliary operations - requires Xfvechalf extension */
{"vfdotp.h",       0, INSN_CLASS_XPULP_FAUXVECHALF, "D,S,T",      MATCH_VFDOTP_H, MASK_VFDOTP_H, match_opcode, 0},
{"vfdotp.r.h",     0, INSN_CLASS_XPULP_FAUXVECHALF, "D,S,T",      MATCH_VFDOTP_R_H, MASK_VFDOTP_R_H, match_opcode, 0},
{"vfdotpex.s.h",   0, INSN_CLASS_XPULP_FAUXVECHALF, "D,S,T", MATCH_VFDOTPEX_S_H, MASK_VFDOTPEX_S_H, match_opcode, 0},
{"vfdotpex.s.r.h", 0, INSN_CLASS_XPULP_FAUXVECHALF, "D,S,T", MATCH_VFDOTPEX_S_R_H, MASK_VFDOTPEX_S_R_H, match_opcode, 0},
{"vfavg.h",        0, INSN_CLASS_XPULP_FAUXVECHALF, "D,S,T",      MATCH_VFAVG_H, MASK_VFAVG_H, match_opcode, 0},
{"vfavg.r.h",      0, INSN_CLASS_XPULP_FAUXVECHALF, "D,S,T",      MATCH_VFAVG_R_H, MASK_VFAVG_R_H, match_opcode, 0},
/* Xfauxalthalf - alternate half-precision auxiliary operations - requires Xfalthalf extension */
{"fmulex.s.ah",  0, INSN_CLASS_XPULP_FAUXALTHALF, "D,S,T",      MATCH_FMULEX_S_AH, MASK_FMULEX_S_AH, match_opcode, 0},
{"fmacex.s.ah",  0, INSN_CLASS_XPULP_FAUXALTHALF, "D,S,T",      MATCH_FMACEX_S_AH, MASK_FMACEX_S_AH, match_opcode, 0},
/* Xfauxvecalthalf - vectorial alternate half-precision auxiliary operations - requires Xfvecalthalf extension */
{"vfdotp.ah",       0, INSN_CLASS_XPULP_FAUXVECALTHALF, "D,S,T",  MATCH_VFDOTP_AH, MASK_VFDOTP_AH, match_opcode, 0},
{"vfdotp.r.ah",     0, INSN_CLASS_XPULP_FAUXVECALTHALF, "D,S,T",  MATCH_VFDOTP_R_AH, MASK_VFDOTP_R_AH, match_opcode, 0},
{"vfdotpex.s.ah",   0, INSN_CLASS_XPULP_FAUXVECALTHALF, "D,S,T", MATCH_VFDOTPEX_S_AH, MASK_VFDOTPEX_S_AH, match_opcode, 0},
{"vfdotpex.s.r.ah", 0, INSN_CLASS_XPULP_FAUXVECALTHALF, "D,S,T", MATCH_VFDOTPEX_S_R_AH, MASK_VFDOTPEX_S_R_AH, match_opcode, 0},
{"vfavg.ah",        0, INSN_CLASS_XPULP_FAUXVECALTHALF, "D,S,T",  MATCH_VFAVG_AH, MASK_VFAVG_AH, match_opcode, 0},
{"vfavg.r.ah",      0, INSN_CLASS_XPULP_FAUXVECALTHALF, "D,S,T",  MATCH_VFAVG_R_AH, MASK_VFAVG_R_AH, match_opcode, 0},
/* Xfauxquarter - quarter-precision auxiliary operations - requires Xfquarter extension */
{"fmulex.s.b",     0, INSN_CLASS_XPULP_FAUXQUARTER, "D,S,T",      MATCH_FMULEX_S_B | MASK_RM, MASK_FMULEX_S_B | MASK_RM, match_opcode, 0},
{"fmulex.s.b",     0, INSN_CLASS_XPULP_FAUXQUARTER, "D,S,T,m",    MATCH_FMULEX_S_B, MASK_FMULEX_S_B, match_opcode, 0},
{"fmacex.s.b",     0, INSN_CLASS_XPULP_FAUXQUARTER, "D,S,T",      MATCH_FMACEX_S_B | MASK_RM, MASK_FMACEX_S_B | MASK_RM, match_opcode, 0},
{"fmacex.s.b",     0, INSN_CLASS_XPULP_FAUXQUARTER, "D,S,T,m",    MATCH_FMACEX_S_B, MASK_FMACEX_S_B, match_opcode, 0},
/* Xfauxvecquarter - vectorial quarter-precision auxiliary operations - requires Xfvecquarter extension */
{"vfdotp.b",       0, INSN_CLASS_XPULP_FAUXVECQUARTER, "D,S,T",   MATCH_VFDOTP_B, MASK_VFDOTP_B, match_opcode, 0},
{"vfdotp.r.b",     0, INSN_CLASS_XPULP_FAUXVECQUARTER, "D,S,T",   MATCH_VFDOTP_R_B, MASK_VFDOTP_R_B, match_opcode, 0},
{"vfdotpex.s.b",   0, INSN_CLASS_XPULP_FAUXVECQUARTER, "D,S,T", MATCH_VFDOTPEX_S_B, MASK_VFDOTPEX_S_B, match_opcode, 0},
{"vfdotpex.s.r.b", 0, INSN_CLASS_XPULP_FAUXVECQUARTER, "D,S,T", MATCH_VFDOTPEX_S_R_B, MASK_VFDOTPEX_S_R_B, match_opcode, 0},
{"vfavg.b",        0, INSN_CLASS_XPULP_FAUXVECQUARTER, "D,S,T",   MATCH_VFAVG_B, MASK_VFAVG_B, match_opcode, 0},
{"vfavg.r.b",      0, INSN_CLASS_XPULP_FAUXVECQUARTER, "D,S,T",   MATCH_VFAVG_R_B, MASK_VFAVG_R_B, match_opcode, 0},




/* Gap 9 specific extensions. */
/* Complex numbers */
{"pv.cplxmul.h.r",      0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S0, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.r.div2", 0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S1, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.r.div4", 0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S2, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.r.div8", 0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULR|MATCH_V_OP_H_VV_S3, MASK_V_OP11,    match_opcode,   0},

{"pv.cplxmul.h.i",      0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S0, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.i.div2", 0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S1, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.i.div4", 0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S2, MASK_V_OP11,    match_opcode,   0},
{"pv.cplxmul.h.i.div8", 0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_CPLXMULI|MATCH_V_OP_H_VV_S3, MASK_V_OP11,    match_opcode,   0},

{"pv.subrotmj.h",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S0, MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div2",  0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S1, MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div4",  0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S2, MASK_V_OP,      match_opcode,   0},
{"pv.subrotmj.h.div8",  0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUBROTMJ|MATCH_V_OP_H_VV_S3, MASK_V_OP,      match_opcode,   0},

{"pv.cplxconj.h",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s",   MATCH_V_OP_CPLXCONJ|MATCH_V_OP_H_VV,    MASK_V_OP2,     match_opcode,   0},

{"pv.add.h.div2",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_ADD_DIV|MATCH_V_OP_H_VV_S1,  MASK_V_OP,      match_opcode,   0},
{"pv.add.h.div4",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_ADD_DIV|MATCH_V_OP_H_VV_S2,  MASK_V_OP,      match_opcode,   0},
{"pv.add.h.div8",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_ADD_DIV|MATCH_V_OP_H_VV_S3,  MASK_V_OP,      match_opcode,   0},

{"pv.sub.h.div2",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUB_DIV|MATCH_V_OP_H_VV_S1,  MASK_V_OP,      match_opcode,   0},
{"pv.sub.h.div4",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUB_DIV|MATCH_V_OP_H_VV_S2,  MASK_V_OP,      match_opcode,   0},
{"pv.sub.h.div8",       0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_SUB_DIV|MATCH_V_OP_H_VV_S3,  MASK_V_OP,      match_opcode,   0},

{"pv.pack.h.h",         0, INSN_CLASS_XPULP_VECT_GAP9,   "d,s,t", MATCH_V_OP_PACKH|MATCH_V_OP_H_VV,       MASK_V_OP,      match_opcode,   0},

/* Single-precision floating-point instruction subset when float registers are
 * mapped on the int reg file */
/* Riscv F with floats mapped on to integer register file */
{"frsr",      0, INSN_CLASS_XPULP_FINX_GAP9,   "d",                MATCH_FRCSR,                    MASK_FRCSR,             match_opcode, 0 },
{"fssr",      0, INSN_CLASS_XPULP_FINX_GAP9,   "s",                MATCH_FSCSR,                    MASK_FSCSR | MASK_RD,   match_opcode, 0 },
{"fssr",      0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",              MATCH_FSCSR,                    MASK_FSCSR,             match_opcode, 0 },
{"frcsr",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d",                MATCH_FRCSR,                    MASK_FRCSR,             match_opcode, 0 },
{"fscsr",     0, INSN_CLASS_XPULP_FINX_GAP9,   "s",                MATCH_FSCSR,                    MASK_FSCSR | MASK_RD,   match_opcode, 0 },
{"fscsr",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",              MATCH_FSCSR,                    MASK_FSCSR,             match_opcode, 0 },
{"frrm",      0, INSN_CLASS_XPULP_FINX_GAP9,   "d",                MATCH_FRRM,                     MASK_FRRM,              match_opcode, 0 },
{"fsrm",      0, INSN_CLASS_XPULP_FINX_GAP9,   "s",                MATCH_FSRM,                     MASK_FSRM | MASK_RD,    match_opcode, 0 },
{"fsrm",      0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",              MATCH_FSRM,                     MASK_FSRM,              match_opcode, 0 },
{"frflags",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d",                MATCH_FRFLAGS,                  MASK_FRFLAGS,           match_opcode, 0 },
{"fsflags",   0, INSN_CLASS_XPULP_FINX_GAP9,   "s",                MATCH_FSFLAGS,                  MASK_FSFLAGS | MASK_RD, match_opcode, 0 },
{"fsflags",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",              MATCH_FSFLAGS,                  MASK_FSFLAGS,           match_opcode, 0 },

/* Unused since we map float to int register file
{"flw",       "32C", "D,Cm(Cc)",  MATCH_C_FLWSP, MASK_C_FLWSP, match_opcode, INSN_ALIAS },
{"flw",       "32C", "CD,Ck(Cs)",  MATCH_C_FLW, MASK_C_FLW, match_opcode, INSN_ALIAS },
{"flw",       "F",   "D,o(s)",  MATCH_FLW, MASK_FLW, match_opcode, 0 },
{"flw",       "F",   "D,A,s",  0, (int) M_FLW, match_never, INSN_MACRO },
{"fsw",       "32C", "CT,CM(Cc)",  MATCH_C_FSWSP, MASK_C_FSWSP, match_opcode, INSN_ALIAS },
{"fsw",       "32C", "CD,Ck(Cs)",  MATCH_C_FSW, MASK_C_FSW, match_opcode, INSN_ALIAS },
{"fsw",       "F",   "T,q(s)",  MATCH_FSW, MASK_FSW, match_opcode, 0 },
{"fsw",       "F",   "T,A,s",  0, (int) M_FSW, match_never, INSN_MACRO },
{"fmv.x.s",   "F",   "d,S",  MATCH_FMV_X_S, MASK_FMV_X_S, match_opcode, 0 },
{"fmv.s.x",   "F",   "D,s",  MATCH_FMV_S_X, MASK_FMV_S_X, match_opcode, 0 },
{"fmv.s",     "F",   "D,U",  MATCH_FSGNJ_S, MASK_FSGNJ_S, match_rs1_eq_rs2, INSN_ALIAS },
*/

{"fneg.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,w",      MATCH_FSGNJN_S,                 MASK_FSGNJN_S,          match_rs1_eq_rs2, INSN_ALIAS },
{"fabs.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,w",      MATCH_FSGNJX_S,                 MASK_FSGNJX_S,          match_rs1_eq_rs2, INSN_ALIAS },
{"fsgnj.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FSGNJ_S,                  MASK_FSGNJ_S,           match_opcode, 0 },
{"fsgnjn.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FSGNJN_S,                 MASK_FSGNJN_S,          match_opcode, 0 },
{"fsgnjx.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FSGNJX_S,                 MASK_FSGNJX_S,          match_opcode, 0 },
{"fadd.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FADD_S | MASK_RM,         MASK_FADD_S | MASK_RM,  match_opcode, 0 },
{"fadd.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,m",  MATCH_FADD_S,                   MASK_FADD_S,            match_opcode, 0 },
{"fsub.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FSUB_S | MASK_RM,         MASK_FSUB_S | MASK_RM,  match_opcode, 0 },
{"fsub.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,m",  MATCH_FSUB_S,                   MASK_FSUB_S,            match_opcode, 0 },
{"fmul.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FMUL_S | MASK_RM,         MASK_FMUL_S | MASK_RM,  match_opcode, 0 },
{"fmul.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,m",  MATCH_FMUL_S,                   MASK_FMUL_S,            match_opcode, 0 },
{"fdiv.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FDIV_S | MASK_RM,         MASK_FDIV_S | MASK_RM,  match_opcode, 0 },
{"fdiv.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,m",  MATCH_FDIV_S,                   MASK_FDIV_S,            match_opcode, 0 },
{"fsqrt.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",      MATCH_FSQRT_S | MASK_RM,        MASK_FSQRT_S | MASK_RM, match_opcode, 0 },
{"fsqrt.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,m",    MATCH_FSQRT_S,                  MASK_FSQRT_S,           match_opcode, 0 },
{"fmin.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FMIN_S,                   MASK_FMIN_S,            match_opcode, 0 },
{"fmax.s",    0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FMAX_S,                   MASK_FMAX_S,            match_opcode, 0 },
{"fmadd.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e",  MATCH_FMADD_S | MASK_RM,        MASK_FMADD_S | MASK_RM, match_opcode, 0 },
{"fmadd.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e,m",MATCH_FMADD_S,                  MASK_FMADD_S,           match_opcode, 0 },
{"fnmadd.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e",  MATCH_FNMADD_S | MASK_RM,       MASK_FNMADD_S| MASK_RM, match_opcode, 0 },
{"fnmadd.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e,m",MATCH_FNMADD_S,                 MASK_FNMADD_S,          match_opcode, 0 },
{"fmsub.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e",  MATCH_FMSUB_S | MASK_RM,        MASK_FMSUB_S | MASK_RM, match_opcode, 0 },
{"fmsub.s",   0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e,m",MATCH_FMSUB_S,                  MASK_FMSUB_S,           match_opcode, 0 },
{"fnmsub.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e",  MATCH_FNMSUB_S | MASK_RM,       MASK_FNMSUB_S| MASK_RM, match_opcode, 0 },
{"fnmsub.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t,e,m",MATCH_FNMSUB_S,                 MASK_FNMSUB_S,          match_opcode, 0 },
{"fcvt.w.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",      MATCH_FCVT_W_S | MASK_RM,       MASK_FCVT_W_S| MASK_RM, match_opcode, 0 },
{"fcvt.w.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,m",    MATCH_FCVT_W_S,                 MASK_FCVT_W_S,          match_opcode, 0 },
{"fcvt.wu.s", 0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",      MATCH_FCVT_WU_S | MASK_RM,      MASK_FCVT_WU_S|MASK_RM, match_opcode, 0 },
{"fcvt.wu.s", 0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,m",    MATCH_FCVT_WU_S,                MASK_FCVT_WU_S,         match_opcode, 0 },
{"fcvt.s.w",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",      MATCH_FCVT_S_W | MASK_RM,       MASK_FCVT_S_W| MASK_RM, match_opcode, 0 },
{"fcvt.s.w",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,m",    MATCH_FCVT_S_W,                 MASK_FCVT_S_W,          match_opcode, 0 },
{"fcvt.s.wu", 0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",      MATCH_FCVT_S_WU | MASK_RM,      MASK_FCVT_S_W| MASK_RM, match_opcode, 0 },
{"fcvt.s.wu", 0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,m",    MATCH_FCVT_S_WU,                MASK_FCVT_S_WU,         match_opcode, 0 },
{"fclass.s",  0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s",      MATCH_FCLASS_S,                 MASK_FCLASS_S,          match_opcode, 0 },
{"feq.s",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FEQ_S,                    MASK_FEQ_S,             match_opcode, 0 },
{"flt.s",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FLT_S,                    MASK_FLT_S,             match_opcode, 0 },
{"fle.s",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d,s,t",    MATCH_FLE_S,                    MASK_FLE_S,             match_opcode, 0 },
{"fgt.s",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d,t,s",    MATCH_FLT_S,                    MASK_FLT_S,             match_opcode, 0 },
{"fge.s",     0, INSN_CLASS_XPULP_FINX_GAP9,   "d,t,s",            MATCH_FLE_S,                    MASK_FLE_S,             match_opcode, 0 },

/* Gap9 FP16 ALT,  Riscv F based but on fp16alt */

{"fmadd.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t,e",  MATCH_FMADD_AH,                 MASK_FMADD_AH,          match_opcode, 0 },
{"fmsub.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t,e",  MATCH_FMSUB_AH,                 MASK_FMSUB_AH,          match_opcode, 0 },
{"fnmsub.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t,e",  MATCH_FNMSUB_AH,                MASK_FNMSUB_AH,         match_opcode, 0 },
{"fnmadd.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t,e",  MATCH_FNMADD_AH,                MASK_FNMADD_AH,         match_opcode, 0 },
{"fadd.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FADD_AH,                  MASK_FADD_AH,           match_opcode, 0 },
{"fsub.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FSUB_AH,                  MASK_FSUB_AH,           match_opcode, 0 },
{"fmul.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FMUL_AH,                  MASK_FMUL_AH,           match_opcode, 0 },
{"fdiv.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FDIV_AH,                  MASK_FDIV_AH,           match_opcode, 0 },
{"fsqrt.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FSQRT_AH,                 MASK_FSQRT_AH,          match_opcode, 0 },
{"fsgnj.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FSGNJ_AH,                 MASK_FSGNJ_AH,          match_opcode, 0 },
{"fsgnjn.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FSGNJN_AH,                MASK_FSGNJN_AH,         match_opcode, 0 },
{"fsgnjx.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FSGNJX_AH,                MASK_FSGNJX_AH,         match_opcode, 0 },
{"fmin.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FMIN_AH,                  MASK_FMIN_AH,           match_opcode, 0 },
{"fmax.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FMAX_AH,                  MASK_FMAX_AH,           match_opcode, 0 },
{"feq.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FEQ_AH,                   MASK_FEQ_AH,            match_opcode, 0 },
{"flt.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FLT_AH,                   MASK_FLT_AH,            match_opcode, 0 },
{"fle.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,t",    MATCH_FLE_AH,                   MASK_FLE_AH,            match_opcode, 0 },
{"fcvt.w.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCVT_W_AH,                MASK_FCVT_W_AH,         match_opcode, 0 },
{"fcvt.wu.ah", 0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCVT_WU_AH,               MASK_FCVT_WU_AH,        match_opcode, 0 },
{"fcvt.ah.w",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCVT_AH_W,                MASK_FCVT_AH_W,         match_opcode, 0 },
{"fcvt.ah.wu", 0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCVT_AH_WU,               MASK_FCVT_AH_W,         match_opcode, 0 },
{"fcvt.s.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCVT_S_AH,                MASK_FCVT_S_AH,         match_opcode, 0 },
{"fcvt.ah.s",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCVT_AH_S,                MASK_FCVT_AH_S,         match_opcode, 0 },

{"fclass.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",      MATCH_FCLASS_AH,                MASK_FCLASS_AH,         match_opcode, 0 },
// pseudos
{"fgt.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,t,s",    MATCH_FLT_AH,                   MASK_FLT_AH,            match_opcode, 0 },
{"fge.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,t,s",    MATCH_FLE_AH,                   MASK_FLE_AH,            match_opcode, 0 },
{"fabs.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,w",      MATCH_FSGNJX_AH,                MASK_FSGNJX_AH,         match_rs1_eq_rs2, INSN_ALIAS },
{"fneg.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,w",      MATCH_FSGNJN_AH,                MASK_FSGNJN_AH,         match_rs1_eq_rs2, INSN_ALIAS },

/* Gap9 FP16,  Riscv F based but on fp16 */
{"fmadd.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e",         MATCH_FMADD_H | MASK_RM,        MASK_FMADD_H | MASK_RM, match_opcode,   0},
{"fmadd.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e,m",       MATCH_FMADD_H,                  MASK_FMADD_H,           match_opcode,   0},
{"fmsub.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e",         MATCH_FMSUB_H | MASK_RM,        MASK_FMSUB_H | MASK_RM, match_opcode,   0},
{"fmsub.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e,m",       MATCH_FMSUB_H,                  MASK_FMSUB_H,           match_opcode,   0},
{"fnmsub.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e",         MATCH_FNMSUB_H | MASK_RM,       MASK_FNMSUB_H|MASK_RM,  match_opcode,   0},
{"fnmsub.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e,m",       MATCH_FNMSUB_H,                 MASK_FNMSUB_H,          match_opcode,   0},
{"fnmadd.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e",         MATCH_FNMADD_H | MASK_RM,       MASK_FNMADD_H|MASK_RM,  match_opcode,   0},
{"fnmadd.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,e,m",       MATCH_FNMADD_H,                 MASK_FNMADD_H,          match_opcode,   0},
{"fadd.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FADD_H | MASK_RM,         MASK_FADD_H | MASK_RM,  match_opcode,   0},
{"fadd.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,m",         MATCH_FADD_H,                   MASK_FADD_H,            match_opcode,   0},
{"fsub.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FSUB_H | MASK_RM,         MASK_FSUB_H | MASK_RM,  match_opcode,   0},
{"fsub.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,m",         MATCH_FSUB_H,                   MASK_FSUB_H,            match_opcode,   0},
{"fmul.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FMUL_H | MASK_RM,         MASK_FMUL_H | MASK_RM,  match_opcode,   0},
{"fmul.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,m",         MATCH_FMUL_H,                   MASK_FMUL_H,            match_opcode,   0},
{"fdiv.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FDIV_H | MASK_RM,         MASK_FDIV_H | MASK_RM,  match_opcode,   0},
{"fdiv.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t,m",         MATCH_FDIV_H,                   MASK_FDIV_H,            match_opcode,   0},
{"fsqrt.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FSQRT_H | MASK_RM,        MASK_FSQRT_H | MASK_RM, match_opcode,   0},
{"fsqrt.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,m",           MATCH_FSQRT_H,                  MASK_FSQRT_H,           match_opcode,   0},
{"fsgnj.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FSGNJ_H,                  MASK_FSGNJ_H,           match_opcode,   0},
{"fsgnjn.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FSGNJN_H,                 MASK_FSGNJN_H,          match_opcode,   0},
{"fsgnjx.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FSGNJX_H,                 MASK_FSGNJX_H,          match_opcode,   0},
{"fmin.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FMIN_H,                   MASK_FMIN_H,            match_opcode,   0},
{"fmax.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FMAX_H,                   MASK_FMAX_H,            match_opcode,   0},
{"feq.h",     0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FEQ_H,                    MASK_FEQ_H,             match_opcode,   0},
{"flt.h",     0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FLT_H,                    MASK_FLT_H,             match_opcode,   0},
{"fle.h",     0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,t",           MATCH_FLE_H,                    MASK_FLE_H,             match_opcode,   0},

{"fcvt.w.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCVT_W_H | MASK_RM,       MASK_FCVT_W_H|MASK_RM,  match_opcode,   0},
{"fcvt.w.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,m",           MATCH_FCVT_W_H,                 MASK_FCVT_W_H,          match_opcode,   0},
{"fcvt.wu.h", 0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCVT_WU_H | MASK_RM,      MASK_FCVT_WU_H|MASK_RM, match_opcode,   0},
{"fcvt.wu.h", 0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,m",           MATCH_FCVT_WU_H,                MASK_FCVT_WU_H,         match_opcode,   0},
{"fcvt.h.w",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCVT_H_W | MASK_RM,       MASK_FCVT_H_W|MASK_RM,  match_opcode,   0},
{"fcvt.h.w",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,m",           MATCH_FCVT_H_W,                 MASK_FCVT_H_W,          match_opcode,   0},
{"fcvt.h.wu", 0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCVT_H_WU | MASK_RM,      MASK_FCVT_H_W|MASK_RM,  match_opcode,   0},
{"fcvt.h.wu", 0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,m",           MATCH_FCVT_H_WU,                MASK_FCVT_H_WU,         match_opcode,   0},
{"fcvt.s.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCVT_S_H,                 MASK_FCVT_S_H,          match_opcode,   0},
{"fcvt.h.s",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCVT_H_S | MASK_RM,       MASK_FCVT_H_S|MASK_RM,  match_opcode,   0},
{"fcvt.h.s",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s,m",           MATCH_FCVT_H_S,                 MASK_FCVT_H_S,          match_opcode,   0},

{"fclass.h",  0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,s",             MATCH_FCLASS_H,                 MASK_FCLASS_H,          match_opcode,   0},

// pseudos
{"fgt.h",     0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,t,s",           MATCH_FLT_H,                    MASK_FLT_H,             match_opcode,  INSN_ALIAS },
{"fge.h",     0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,t,s",           MATCH_FLE_H,                    MASK_FLE_H,             match_opcode,  INSN_ALIAS },
{"fabs.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,w",             MATCH_FSGNJX_H,                 MASK_FSGNJX_H,          match_rs1_eq_rs2,   INSN_ALIAS },
{"fneg.h",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,w",             MATCH_FSGNJN_H,                 MASK_FSGNJN_H,          match_rs1_eq_rs2,   INSN_ALIAS },


/* Gap9 FP16 ALT <-> Fp16 */
{"fcvt.h.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s"  ,          MATCH_FCVT_H_AH | MASK_RM,      MASK_FCVT_H_AH|MASK_RM, match_opcode, 0 },
{"fcvt.h.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s,m",          MATCH_FCVT_H_AH,                MASK_FCVT_H_AH,         match_opcode, 0 },
{"fcvt.ah.h",   0, INSN_CLASS_XPULP_FHALF_GAP9,  "d,s",            MATCH_FCVT_AH_H,                MASK_FCVT_AH_H,         match_opcode, 0 },

/* Gap9 FP16 ALT,  Fp16Alt, Fp16Alt => Fp32 */
{"fmulex.s.ah", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_FMULEX_S_AH,              MASK_FMULEX_S_AH,       match_opcode, 0},
{"fmacex.s.ah", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_FMACEX_S_AH,              MASK_FMACEX_S_AH,       match_opcode, 0},

/* Gap9 SIMD2 FP16ALT */
{"vfadd.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFADD_AH,                 MASK_VFADD_AH,          match_opcode, 0},
{"vfadd.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFADD_R_AH,               MASK_VFADD_R_AH,        match_opcode, 0},
{"vfsub.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSUB_AH,                 MASK_VFSUB_AH,          match_opcode, 0},
{"vfsub.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSUB_R_AH,               MASK_VFSUB_R_AH,        match_opcode, 0},
{"vfmul.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMUL_AH,                 MASK_VFMUL_AH,          match_opcode, 0},
{"vfmul.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMUL_R_AH,               MASK_VFMUL_R_AH,        match_opcode, 0},
// {"vfdiv.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",        MATCH_VFDIV_AH,                 MASK_VFDIV_AH,          match_opcode, 0},
// {"vfdiv.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",        MATCH_VFDIV_R_AH,               MASK_VFDIV_R_AH,        match_opcode, 0},
{"vfmin.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMIN_AH,                 MASK_VFMIN_AH,          match_opcode, 0},
{"vfmin.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMIN_R_AH,               MASK_VFMIN_R_AH,        match_opcode, 0},
{"vfmax.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAX_AH,                 MASK_VFMAX_AH,          match_opcode, 0},
{"vfmax.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAX_R_AH,               MASK_VFMAX_R_AH,        match_opcode, 0},
// {"vfsqrt.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",          MATCH_VFSQRT_AH,                MASK_VFSQRT_AH,         match_opcode, 0},
{"vfmac.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAC_AH,                 MASK_VFMAC_AH,          match_opcode, 0},
{"vfmac.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAC_R_AH,               MASK_VFMAC_R_AH,        match_opcode, 0},
{"vfmre.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMRE_AH,                 MASK_VFMRE_AH,          match_opcode, 0},
{"vfmre.r.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMRE_R_AH,               MASK_VFMRE_R_AH,        match_opcode, 0},
{"vfclass.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCLASS_AH_GAP9,          MASK_VFCLASS_AH_GAP9,   match_opcode, 0},
{"vfsgnj.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJ_AH,                MASK_VFSGNJ_AH,         match_opcode, 0},
{"vfsgnj.r.ah", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJ_R_AH,              MASK_VFSGNJ_R_AH,       match_opcode, 0},
{"vfsgnjn.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJN_AH,               MASK_VFSGNJN_AH,        match_opcode, 0},
{"vfsgnjn.r.ah",0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJN_R_AH,             MASK_VFSGNJN_R_AH,      match_opcode, 0},
{"vfsgnjx.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJX_AH,               MASK_VFSGNJX_AH,        match_opcode, 0},
{"vfsgnjx.r.ah",0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJX_R_AH,             MASK_VFSGNJX_R_AH,      match_opcode, 0},
{"vfeq.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFEQ_AH,                  MASK_VFEQ_AH,           match_opcode, 0},
{"vfeq.r.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFEQ_R_AH,                MASK_VFEQ_R_AH,         match_opcode, 0},
{"vfne.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFNE_AH,                  MASK_VFNE_AH,           match_opcode, 0},
{"vfne.r.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFNE_R_AH,                MASK_VFNE_R_AH,         match_opcode, 0},
{"vflt.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLT_AH,                  MASK_VFLT_AH,           match_opcode, 0},
{"vflt.r.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLT_R_AH,                MASK_VFLT_R_AH,         match_opcode, 0},
{"vfge.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGE_AH,                  MASK_VFGE_AH,           match_opcode, 0},
{"vfge.r.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGE_R_AH,                MASK_VFGE_R_AH,         match_opcode, 0},
{"vfle.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLE_AH,                  MASK_VFLE_AH,           match_opcode, 0},
{"vfle.r.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLE_R_AH,                MASK_VFLE_R_AH,         match_opcode, 0},
{"vfgt.ah",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGT_AH,                  MASK_VFGT_AH,           match_opcode, 0},
{"vfgt.r.ah",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGT_R_AH,                MASK_VFGT_R_AH,         match_opcode, 0},
{"vfcpka.ah.s", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFCPKA_AH_S,              MASK_VFCPKA_AH_S,       match_opcode, 0},

{"vfcvt.x.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_X_AH,               MASK_VFCVT_X_AH,        match_opcode, 0},
{"vfcvt.xu.ah", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_XU_AH,              MASK_VFCVT_XU_AH,       match_opcode, 0},
{"vfcvt.ah.x",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_AH_X,               MASK_VFCVT_AH_X,        match_opcode, 0},
{"vfcvt.ah.xu", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_AH_XU,              MASK_VFCVT_AH_XU,       match_opcode, 0},

/* Pseudos */
{"vfabs.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,w",           MATCH_VFSGNJX_AH,               MASK_VFSGNJX_AH,        match_rs1_eq_rs2,   INSN_ALIAS },
{"vfneg.ah",    0, INSN_CLASS_XPULP_FHALF_GAP9,   "d,w",           MATCH_VFSGNJN_AH,               MASK_VFSGNJN_AH,        match_rs1_eq_rs2,   INSN_ALIAS },


/* Gap9 SIMD2 FP16 */
{"vfadd.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFADD_H,                  MASK_VFADD_H,           match_opcode, 0},
{"vfadd.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFADD_R_H,                MASK_VFADD_R_H,         match_opcode, 0},
{"vfsub.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSUB_H,                  MASK_VFSUB_H,           match_opcode, 0},
{"vfsub.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSUB_R_H,                MASK_VFSUB_R_H,         match_opcode, 0},
{"vfmul.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMUL_H,                  MASK_VFMUL_H,           match_opcode, 0},
{"vfmul.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMUL_R_H,                MASK_VFMUL_R_H,         match_opcode, 0},
// {"vfdiv.h",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFDIV_H,                  MASK_VFDIV_H,           match_opcode, 0},
// {"vfdiv.r.h",0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFDIV_R_H,                MASK_VFDIV_R_H,         match_opcode, 0},
{"vfmin.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMIN_H,                  MASK_VFMIN_H,           match_opcode, 0},
{"vfmin.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMIN_R_H,                MASK_VFMIN_R_H,         match_opcode, 0},
{"vfmax.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAX_H,                  MASK_VFMAX_H,           match_opcode, 0},
{"vfmax.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAX_R_H,                MASK_VFMAX_R_H,         match_opcode, 0},
// {"vfsqrt.h",         0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",     MATCH_VFSQRT_H,                 MASK_VFSQRT_H,          match_opcode, 0},
{"vfmac.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAC_H,                  MASK_VFMAC_H,           match_opcode, 0},
{"vfmac.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMAC_R_H,                MASK_VFMAC_R_H,         match_opcode, 0},
{"vfmre.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMRE_H,                  MASK_VFMRE_H,           match_opcode, 0},
{"vfmre.r.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFMRE_R_H,                MASK_VFMRE_R_H,         match_opcode, 0},
{"vfclass.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCLASS_H_GAP9,           MASK_VFCLASS_H_GAP9,    match_opcode, 0},
{"vfsgnj.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJ_H,                 MASK_VFSGNJ_H,          match_opcode, 0},
{"vfsgnj.r.h",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJ_R_H,               MASK_VFSGNJ_R_H,        match_opcode, 0},
{"vfsgnjn.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJN_H,                MASK_VFSGNJN_H,         match_opcode, 0},
{"vfsgnjn.r.h", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJN_R_H,              MASK_VFSGNJN_R_H,       match_opcode, 0},
{"vfsgnjx.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJX_H,                MASK_VFSGNJX_H,         match_opcode, 0},
{"vfsgnjx.r.h", 0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFSGNJX_R_H,              MASK_VFSGNJX_R_H,       match_opcode, 0},
{"vfeq.h",      0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFEQ_H,                   MASK_VFEQ_H,            match_opcode, 0},
{"vfeq.r.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFEQ_R_H,                 MASK_VFEQ_R_H,          match_opcode, 0},
{"vfne.h",      0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFNE_H,                   MASK_VFNE_H,            match_opcode, 0},
{"vfne.r.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFNE_R_H,                 MASK_VFNE_R_H,          match_opcode, 0},
{"vflt.h",      0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLT_H,                   MASK_VFLT_H,            match_opcode, 0},
{"vflt.r.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLT_R_H,                 MASK_VFLT_R_H,          match_opcode, 0},
{"vfge.h",      0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGE_H,                   MASK_VFGE_H,            match_opcode, 0},
{"vfge.r.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGE_R_H,                 MASK_VFGE_R_H,          match_opcode, 0},
{"vfle.h",      0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLE_H,                   MASK_VFLE_H,            match_opcode, 0},
{"vfle.r.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFLE_R_H,                 MASK_VFLE_R_H,          match_opcode, 0},
{"vfgt.h",      0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGT_H,                   MASK_VFGT_H,            match_opcode, 0},
{"vfgt.r.h",    0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFGT_R_H,                 MASK_VFGT_R_H,          match_opcode, 0},
{"vfcpka.h.s",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s,t",           MATCH_VFCPKA_H_S,               MASK_VFCPKA_H_S,        match_opcode, 0},

{"vfcvt.x.h",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_X_H,                MASK_VFCVT_X_H,         match_opcode, 0},
{"vfcvt.xu.h",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_XU_H,               MASK_VFCVT_XU_H,        match_opcode, 0},
{"vfcvt.h.x",   0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_H_X,                MASK_VFCVT_H_X,         match_opcode, 0},
{"vfcvt.h.xu",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_H_XU,               MASK_VFCVT_H_XU,        match_opcode, 0},

/* Pseudos */
{"vfabs.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,w",             MATCH_VFSGNJX_H,                MASK_VFSGNJX_H,         match_rs1_eq_rs2,   INSN_ALIAS },
{"vfneg.h",     0, INSN_CLASS_XPULP_FHALF_GAP9, "d,w",             MATCH_VFSGNJN_H,                MASK_VFSGNJN_H,         match_rs1_eq_rs2,   INSN_ALIAS },


/* SIMD2 FP16 <-> FP16ALT */
{"vfcvt.h.ah",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_H_AH,               MASK_VFCVT_H_AH,        match_opcode, 0},
{"vfcvt.ah.h",  0, INSN_CLASS_XPULP_FHALF_GAP9, "d,s",             MATCH_VFCVT_AH_H,               MASK_VFCVT_AH_H,        match_opcode, 0},


/***********************************************************************************************************************/
/**************************************     End of Gap9              ***************************************************/
/***********************************************************************************************************************/

/* Terminate the list.  */
{0, 0, INSN_CLASS_NONE, 0, 0, 0, 0, 0}
};

/* Instruction format for .insn directive.  */
const struct riscv_opcode riscv_insn_types[] =
{
/* name, xlen, isa,          operands, match, mask,    match_func, pinfo.  */
{"r",       0, INSN_CLASS_I,  "O4,F3,F7,d,s,t",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,D,s,t",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,d,S,t",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,D,S,t",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,d,s,T",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,D,s,T",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,d,S,T",     0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F7,D,S,T",     0,    0,  match_opcode, 0 },

{"r",       0, INSN_CLASS_I,  "O4,F3,F2,d,s,t,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,s,t,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,S,t,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,S,t,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,s,T,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,s,T,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,S,T,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,S,T,r",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,s,t,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,s,t,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,S,t,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,S,t,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,s,T,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,s,T,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,d,S,T,R",   0,    0,  match_opcode, 0 },
{"r",       0, INSN_CLASS_F,  "O4,F3,F2,D,S,T,R",   0,    0,  match_opcode, 0 },

{"r4",      0, INSN_CLASS_I,  "O4,F3,F2,d,s,t,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,s,t,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,S,t,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,S,t,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,s,T,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,s,T,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,S,T,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,S,T,r",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,s,t,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,s,t,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,S,t,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,S,t,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,s,T,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,s,T,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,d,S,T,R",   0,    0,  match_opcode, 0 },
{"r4",      0, INSN_CLASS_F,  "O4,F3,F2,D,S,T,R",   0,    0,  match_opcode, 0 },

{"i",       0, INSN_CLASS_I,  "O4,F3,d,s,j",        0,    0,  match_opcode, 0 },
{"i",       0, INSN_CLASS_F,  "O4,F3,D,s,j",        0,    0,  match_opcode, 0 },
{"i",       0, INSN_CLASS_F,  "O4,F3,d,S,j",        0,    0,  match_opcode, 0 },
{"i",       0, INSN_CLASS_F,  "O4,F3,D,S,j",        0,    0,  match_opcode, 0 },

{"i",       0, INSN_CLASS_I,  "O4,F3,d,o(s)",       0,    0,  match_opcode, 0 },
{"i",       0, INSN_CLASS_F,  "O4,F3,D,o(s)",       0,    0,  match_opcode, 0 },

{"s",       0, INSN_CLASS_I,  "O4,F3,t,q(s)",       0,    0,  match_opcode, 0 },
{"s",       0, INSN_CLASS_F,  "O4,F3,T,q(s)",       0,    0,  match_opcode, 0 },

{"sb",      0, INSN_CLASS_I,  "O4,F3,s,t,p",        0,    0,  match_opcode, 0 },
{"sb",      0, INSN_CLASS_F,  "O4,F3,S,t,p",        0,    0,  match_opcode, 0 },
{"sb",      0, INSN_CLASS_F,  "O4,F3,s,T,p",        0,    0,  match_opcode, 0 },
{"sb",      0, INSN_CLASS_F,  "O4,F3,S,T,p",        0,    0,  match_opcode, 0 },

{"b",      0, INSN_CLASS_I,  "O4,F3,s,t,p",        0,    0,  match_opcode, 0 },
{"b",      0, INSN_CLASS_F,  "O4,F3,S,t,p",        0,    0,  match_opcode, 0 },
{"b",      0, INSN_CLASS_F,  "O4,F3,s,T,p",        0,    0,  match_opcode, 0 },
{"b",      0, INSN_CLASS_F,  "O4,F3,S,T,p",        0,    0,  match_opcode, 0 },

{"u",       0, INSN_CLASS_I,  "O4,d,u",             0,    0,  match_opcode, 0 },
{"u",       0, INSN_CLASS_F,  "O4,D,u",             0,    0,  match_opcode, 0 },

{"uj",      0, INSN_CLASS_I,  "O4,d,a",             0,    0,  match_opcode, 0 },
{"uj",      0, INSN_CLASS_F,  "O4,D,a",             0,    0,  match_opcode, 0 },

{"j",      0, INSN_CLASS_I,  "O4,d,a",             0,    0,  match_opcode, 0 },
{"j",      0, INSN_CLASS_F,  "O4,D,a",             0,    0,  match_opcode, 0 },

{"cr",      0, INSN_CLASS_C,  "O2,CF4,d,CV",        0,    0,  match_opcode, 0 },
{"cr",      0, INSN_CLASS_F_AND_C,  "O2,CF4,D,CV",        0,    0,  match_opcode, 0 },
{"cr",      0, INSN_CLASS_F_AND_C,  "O2,CF4,d,CT",        0,    0,  match_opcode, 0 },
{"cr",      0, INSN_CLASS_F_AND_C,  "O2,CF4,D,CT",        0,    0,  match_opcode, 0 },

{"ci",      0, INSN_CLASS_C,  "O2,CF3,d,Co",        0,    0,  match_opcode, 0 },
{"ci",      0, INSN_CLASS_F_AND_C,  "O2,CF3,D,Co",        0,    0,  match_opcode, 0 },

{"ciw",     0, INSN_CLASS_C,  "O2,CF3,Ct,C8",       0,    0,  match_opcode, 0 },
{"ciw",     0, INSN_CLASS_F_AND_C,  "O2,CF3,CD,C8",       0,    0,  match_opcode, 0 },

{"ca",      0, INSN_CLASS_C,  "O2,CF6,CF2,Cs,Ct",   0,    0,  match_opcode, 0 },
{"ca",      0, INSN_CLASS_F_AND_C,  "O2,CF6,CF2,CS,Ct",   0,    0,  match_opcode, 0 },
{"ca",      0, INSN_CLASS_F_AND_C,  "O2,CF6,CF2,Cs,CD",   0,    0,  match_opcode, 0 },
{"ca",      0, INSN_CLASS_F_AND_C,  "O2,CF6,CF2,CS,CD",   0,    0,  match_opcode, 0 },

{"cb",      0, INSN_CLASS_C,  "O2,CF3,Cs,Cp",       0,    0,  match_opcode, 0 },
{"cb",      0, INSN_CLASS_F_AND_C,  "O2,CF3,CS,Cp",       0,    0,  match_opcode, 0 },

{"cj",      0, INSN_CLASS_C,  "O2,CF3,Ca",          0,    0,  match_opcode, 0 },
/* Terminate the list.  */
{0, 0, INSN_CLASS_NONE, 0, 0, 0, 0, 0}
};
