//===-- GCNProcessors.td - GCN Processor definitions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The code produced for "generic" is only useful for tests and cannot
// reasonably be expected to execute on any particular target.
def : ProcessorModel<"generic", NoSchedModel,
  [FeatureWavefrontSize64, FeatureGDS, FeatureGWS]
>;

def : ProcessorModel<"generic-hsa", NoSchedModel,
  [FeatureWavefrontSize64, FeatureGDS, FeatureGWS, FeatureFlatAddressSpace]
>;

//===----------------------------------------------------------------------===//
// GCN GFX10.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1010", GFX10SpeedModel,
  FeatureISAVersion10_1_0.Features
>;

def : ProcessorModel<"gfx1011", GFX10SpeedModel,
  FeatureISAVersion10_1_1.Features
>;

def : ProcessorModel<"gfx1012", GFX10SpeedModel,
  FeatureISAVersion10_1_2.Features
>;

def : ProcessorModel<"gfx1013", GFX10SpeedModel,
  FeatureISAVersion10_1_3.Features
>;

def : ProcessorModel<"gfx1030", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1031", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1032", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1033", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1034", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1035", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

def : ProcessorModel<"gfx1036", GFX10SpeedModel,
  FeatureISAVersion10_3_0.Features
>;

//===----------------------------------------------------------------------===//
// GCN GFX11.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1100", GFX11SpeedModel,
  FeatureISAVersion11_0_0.Features
>;
def : ProcessorModel<"r1000", GFX11SpeedModel,
  FeatureISAVersion11_0_0.Features
>;

def : ProcessorModel<"gfx1101", GFX11SpeedModel,
  FeatureISAVersion11_0_1.Features
>;

def : ProcessorModel<"gfx1102", GFX11SpeedModel,
  FeatureISAVersion11_0_2.Features
>;

def : ProcessorModel<"gfx1103", GFX11SpeedModel,
  FeatureISAVersion11_0_3.Features
>;

def : ProcessorModel<"gfx1150", GFX11SpeedModel,
  FeatureISAVersion11_5_0.Features
>;

def : ProcessorModel<"gfx1151", GFX11SpeedModel,
  FeatureISAVersion11_5_1.Features
>;

//===----------------------------------------------------------------------===//
// GCN GFX12.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"gfx1200", GFX12SpeedModel,
  FeatureISAVersion12.Features
>;

def : ProcessorModel<"gfx1201", GFX12SpeedModel,
  FeatureISAVersion12.Features
>;
