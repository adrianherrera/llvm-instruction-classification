//===-- InstructionClassification.cpp - Halstead complexity ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Classifies LLVM instructions.
///
/// These categories come from the LLVM language reference manual,
/// http://llvm.org/docs/LangRef.html#instruction-reference
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "instruction_classification"

namespace {

class InstructionClassification : public FunctionPass {
private:
  Function *F;
  SmallVector<const Instruction *, 12> TermOps;
  SmallVector<const Instruction *, 12> UnaryOps;
  SmallVector<const Instruction *, 12> BinaryOps;
  SmallVector<const Instruction *, 12> FloatBinaryOps;
  SmallVector<const Instruction *, 12> BitwiseBinaryOps;
  SmallVector<const Instruction *, 12> VectorOps;
  SmallVector<const Instruction *, 12> AggregateOps;
  SmallVector<const Instruction *, 12> MemAccessAndAddrOps;
  SmallVector<const Instruction *, 12> ConvOps;
  SmallVector<const Instruction *, 12> OtherOps;

public:
  static char ID;
  InstructionClassification() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &) const override;
  void print(raw_ostream &, const Module *) const override;
  bool runOnFunction(Function &) override;
};

} // anonymous namespace

char InstructionClassification::ID = 0;

void InstructionClassification::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

void InstructionClassification::print(raw_ostream &OS, const Module *) const {
  size_t NumTermOps = this->TermOps.size();
  size_t NumUnaryOps = this->UnaryOps.size();
  size_t NumBinaryOps = this->BinaryOps.size();
  size_t NumFloatBinaryOps = this->FloatBinaryOps.size();
  size_t NumBitwiseBinaryOps = this->BitwiseBinaryOps.size();
  size_t NumVectorOps = this->VectorOps.size();
  size_t NumAggregateOps = this->AggregateOps.size();
  size_t NumMemAccessAndAddrOps = this->MemAccessAndAddrOps.size();
  size_t NumConvOps = this->ConvOps.size();
  size_t NumOtherOps = this->OtherOps.size();

  OS << "  # terminator operations: " << NumTermOps << '\n';
  OS << "  # unary operations: " << NumUnaryOps << '\n';
  OS << "  # binary operations: " << NumBinaryOps << '\n';
  OS << "  # float binary operations: " << NumFloatBinaryOps << '\n';
  OS << "  # bitwise binary operations: " << NumBitwiseBinaryOps << '\n';
  OS << "  # vector operations: " << NumVectorOps << '\n';
  OS << "  # aggregate operations: " << NumAggregateOps << '\n';
  OS << "  # memory access and addressing operations: "
     << NumMemAccessAndAddrOps << '\n';
  OS << "  # conversion operations: " << NumConvOps << '\n';
  OS << "  # other operations: " << NumOtherOps << '\n';
}

bool InstructionClassification::runOnFunction(Function &F) {
  this->F = &F;

  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    switch (I->getOpcode()) {
    case Instruction::Ret:
    case Instruction::Br:
    case Instruction::Switch:
    case Instruction::IndirectBr:
    case Instruction::Invoke:
    case Instruction::CallBr:
    case Instruction::Resume:
    case Instruction::CatchSwitch:
    case Instruction::CatchRet:
    case Instruction::CleanupRet:
    case Instruction::Unreachable:
      this->TermOps.push_back(&*I);
      break;
    case Instruction::FNeg:
      this->UnaryOps.push_back(&*I);
      break;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      this->BinaryOps.push_back(&*I);
      break;
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FRem:
    case Instruction::FDiv:
      this->FloatBinaryOps.push_back(&*I);
      break;
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      this->BitwiseBinaryOps.push_back(&*I);
      break;
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
      this->VectorOps.push_back(&*I);
      break;
    case Instruction::ExtractValue:
    case Instruction::InsertValue:
      this->AggregateOps.push_back(&*I);
      break;
    case Instruction::Alloca:
    case Instruction::Load:
    case Instruction::Store:
    case Instruction::Fence:
    case Instruction::AtomicCmpXchg:
    case Instruction::AtomicRMW:
    case Instruction::GetElementPtr:
      this->MemAccessAndAddrOps.push_back(&*I);
      break;
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
      this->ConvOps.push_back(&*I);
      break;
    default:
      this->OtherOps.push_back(&*I);
    }
  }

  return false;
}

static RegisterPass<InstructionClassification>
    X("instruction-classification", "Classify LLVM instructions", false, false);

static void registerInstructionClassification(const PassManagerBuilder &,
                                              legacy::PassManagerBase &PM) {
  PM.add(new InstructionClassification());
}

static RegisterStandardPasses
    RegisterInstructionClassification(PassManagerBuilder::EP_OptimizerLast,
                                      registerInstructionClassification);

static RegisterStandardPasses RegisterInstructionClassification0(
    PassManagerBuilder::EP_EnabledOnOptLevel0,
    registerInstructionClassification);
