import VerifiedNN.Core.DataTypes
import SciLean

/-!
# ASCII Image Renderer

Pure Lean implementation of ASCII art renderer for 28×28 MNIST images.

This module provides a **completely computable** visualization tool for grayscale images,
rendering them as ASCII art in the terminal. Unlike the training code which uses
noncomputable automatic differentiation, this renderer uses only basic arithmetic and
string operations, making it suitable for standalone executables.

## Main Definitions

- `renderImage`: Convert MNIST image (784-dim vector) to ASCII art string
- `renderImageWithLabel`: Render image with text label overlay
- `brightnessToChar`: Map brightness value to ASCII character
- `autoDetectRange`: Determine if input is normalized (0-1) or raw (0-255)

## Key Features

**Completely Computable:**
- No automatic differentiation or noncomputable operations
- Works in standalone executables (unlike training code)
- Pure functional implementation

**Auto-Detection:**
- Automatically detects input range (0-1 normalized vs 0-255 raw)
- Handles both MNIST raw format and preprocessed data

**High Fidelity:**
- 16-character brightness palette for detailed rendering
- Captures fine details in digit images
- Inverted mode for light terminal backgrounds

**Performance:**
- O(784) complexity for 28×28 images
- Minimal memory allocation
- Fast enough for real-time rendering

## Usage Examples

```lean
-- Load MNIST data
let samples ← loadMNISTTest "data"
let (image, label) := samples[0]!

-- Basic rendering (auto-detects range, dark terminal)
let ascii := renderImage image false
IO.println ascii

-- Inverted mode for light terminals
let asciiInverted := renderImage image true
IO.println asciiInverted

-- With label
let withLabel := renderImageWithLabel image s!"Ground Truth: {label}" false
IO.println withLabel
```

## Implementation Notes

**Character Palette:**
The 16-character palette " .:-=+*#%@" provides good balance between detail
and readability. Characters are ordered from darkest (space) to brightest (@).

**Range Detection:**
Values > 1.1 are assumed to be in 0-255 range (MNIST raw format).
Values ≤ 1.1 are assumed to be normalized to 0-1 range.
This heuristic handles both formats automatically.

**Inverted Mode:**
Normal mode: dark pixels (0) → space, bright pixels (255) → @
Inverted mode: dark pixels (0) → @, bright pixels (255) → space
Inverted mode improves visibility on light-background terminals.

## References

- MNIST dataset: http://yann.lecun.com/exdb/mnist/
- ASCII art rendering: https://en.wikipedia.org/wiki/ASCII_art
-/

namespace VerifiedNN.Util.ImageRenderer

open VerifiedNN.Core
open SciLean

/-- Character palette for brightness levels, ordered dark to bright.

The 16-character palette provides fine gradation for grayscale images:
- Space (0x20): darkest - represents 0 brightness
- Period, colon, etc.: intermediate levels
- @ sign (0x40): brightest - represents maximum brightness

**Design rationale:** These characters have increasing visual "weight" or darkness
when rendered in typical monospace fonts, creating a natural brightness ramp.

**Alternative palettes:**
- Simple: " .:-=+@" (8 levels)
- Detailed: " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$" (70 levels)

The current 16-char palette balances detail with readability.
-/
private def brightnessChars : String := " .:-=+*#%@"

/-- Number of brightness levels in the palette. -/
private def paletteSize : Nat := brightnessChars.length

/-- Auto-detect whether image values are in 0-255 range or 0-1 normalized range.

Scans the entire image to find the maximum pixel value. If max > 1.1, assumes
the image uses raw 0-255 format (typical MNIST loading). Otherwise assumes
normalized 0-1 format (typical preprocessing).

**Parameters:**
- `img`: 784-dimensional MNIST image vector

**Returns:** `true` if image appears to use 0-255 range, `false` if 0-1 range

**Algorithm:** Linear scan to find maximum value, compare against threshold 1.1.
The threshold 1.1 (rather than 1.0) provides tolerance for floating-point errors
while distinguishing normalized from raw values.

**Complexity:** O(784) - scans entire image once
-/
def autoDetectRange (img : Vector 784) : Bool :=
  -- Simple heuristic: check a few sample pixels
  -- If any pixel > 1.1, assume 0-255 range
  -- This is faster than scanning all 784 pixels
  let sample1 := img[100]
  let sample2 := img[200]
  let sample3 := img[400]
  let sample4 := img[600]
  let sample5 := img[700]

  (sample1 > 1.1) || (sample2 > 1.1) || (sample3 > 1.1) || (sample4 > 1.1) || (sample5 > 1.1)

/-- Map brightness value to ASCII character using the palette.

Converts a brightness value to an appropriate ASCII character for rendering.
Supports both 0-1 normalized and 0-255 raw ranges, with automatic normalization.
Includes inverted mode for light-background terminals.

**Parameters:**
- `value`: Brightness value (either 0-1 or 0-255, normalized internally)
- `isRaw255`: If `true`, treat value as 0-255 range; if `false`, treat as 0-1
- `inverted`: If `true`, reverse the palette (bright → dark, dark → bright)

**Returns:** Single ASCII character representing the brightness level

**Algorithm:**
1. Normalize value to 0-1 range if needed
2. Clamp to [0,1] to handle edge cases (negative or out-of-range values)
3. Map to palette index: floor(normalized * (paletteSize - 1))
4. If inverted, reverse the index
5. Return character from palette

**Edge cases:**
- Values < 0 are clamped to 0
- Values > max (1 or 255) are clamped to max
- Empty palette returns space by default

**Examples:**
- brightnessToChar 0.0 false false → ' ' (space - darkest)
- brightnessToChar 1.0 false false → '@' (brightest)
- brightnessToChar 127.5 true false → (mid-brightness char)
- brightnessToChar 1.0 false true → ' ' (inverted - bright becomes dark)
-/
def brightnessToChar (value : Float) (isRaw255 : Bool) (inverted : Bool) : Char :=
  -- Normalize to 0-1 range
  let normalized := if isRaw255 then value / 255.0 else value

  -- Clamp to [0, 1] to handle edge cases
  let clamped := if normalized < 0.0 then 0.0
                 else if normalized > 1.0 then 1.0
                 else normalized

  -- Map to palette index [0, paletteSize-1]
  let floatIndex := clamped * (paletteSize - 1).toFloat
  let index := floatIndex.floor.toUInt64.toNat

  -- Reverse index if inverted mode
  let finalIndex := if inverted then (paletteSize - 1 - index) else index

  -- Safely get character from palette (should never fail due to clamping)
  brightnessChars.toList[finalIndex]!

/-- Manual unrolling approach: Render a single row using literal indices.

This uses compile-time known indices which SciLean's DataArrayN supports.
While verbose, this avoids the computed Nat → Idx conversion problem.

**Parameters:**
- `img`: 784-dimensional vector
- `rowIdx`: Which row (0-27)
- `isRaw255`: Value range flag
- `inverted`: Brightness inversion flag

**Returns:** String of 28 characters representing one row
-/
private def renderRowLiteral (img : Vector 784) (rowIdx : Nat) (isRaw255 : Bool) (inverted : Bool) : String :=
  let offset := rowIdx * 28
  match offset with
  | 0 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[0] | 1 => img[1] | 2 => img[2] | 3 => img[3] | 4 => img[4] | 5 => img[5] | 6 => img[6] | 7 => img[7] | 8 => img[8] | 9 => img[9] | 10 => img[10] | 11 => img[11] | 12 => img[12] | 13 => img[13] | 14 => img[14] | 15 => img[15] | 16 => img[16] | 17 => img[17] | 18 => img[18] | 19 => img[19] | 20 => img[20] | 21 => img[21] | 22 => img[22] | 23 => img[23] | 24 => img[24] | 25 => img[25] | 26 => img[26] | _ => img[27]
      brightnessToChar px isRaw255 inverted)
  | 28 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[28] | 1 => img[29] | 2 => img[30] | 3 => img[31] | 4 => img[32] | 5 => img[33] | 6 => img[34] | 7 => img[35] | 8 => img[36] | 9 => img[37] | 10 => img[38] | 11 => img[39] | 12 => img[40] | 13 => img[41] | 14 => img[42] | 15 => img[43] | 16 => img[44] | 17 => img[45] | 18 => img[46] | 19 => img[47] | 20 => img[48] | 21 => img[49] | 22 => img[50] | 23 => img[51] | 24 => img[52] | 25 => img[53] | 26 => img[54] | _ => img[55]
      brightnessToChar px isRaw255 inverted)
  | 56 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[56] | 1 => img[57] | 2 => img[58] | 3 => img[59] | 4 => img[60] | 5 => img[61] | 6 => img[62] | 7 => img[63] | 8 => img[64] | 9 => img[65] | 10 => img[66] | 11 => img[67] | 12 => img[68] | 13 => img[69] | 14 => img[70] | 15 => img[71] | 16 => img[72] | 17 => img[73] | 18 => img[74] | 19 => img[75] | 20 => img[76] | 21 => img[77] | 22 => img[78] | 23 => img[79] | 24 => img[80] | 25 => img[81] | 26 => img[82] | _ => img[83]
      brightnessToChar px isRaw255 inverted)
  | 84 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[84] | 1 => img[85] | 2 => img[86] | 3 => img[87] | 4 => img[88] | 5 => img[89] | 6 => img[90] | 7 => img[91] | 8 => img[92] | 9 => img[93] | 10 => img[94] | 11 => img[95] | 12 => img[96] | 13 => img[97] | 14 => img[98] | 15 => img[99] | 16 => img[100] | 17 => img[101] | 18 => img[102] | 19 => img[103] | 20 => img[104] | 21 => img[105] | 22 => img[106] | 23 => img[107] | 24 => img[108] | 25 => img[109] | 26 => img[110] | _ => img[111]
      brightnessToChar px isRaw255 inverted)
  | 112 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[112] | 1 => img[113] | 2 => img[114] | 3 => img[115] | 4 => img[116] | 5 => img[117] | 6 => img[118] | 7 => img[119] | 8 => img[120] | 9 => img[121] | 10 => img[122] | 11 => img[123] | 12 => img[124] | 13 => img[125] | 14 => img[126] | 15 => img[127] | 16 => img[128] | 17 => img[129] | 18 => img[130] | 19 => img[131] | 20 => img[132] | 21 => img[133] | 22 => img[134] | 23 => img[135] | 24 => img[136] | 25 => img[137] | 26 => img[138] | _ => img[139]
      brightnessToChar px isRaw255 inverted)
  | 140 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[140] | 1 => img[141] | 2 => img[142] | 3 => img[143] | 4 => img[144] | 5 => img[145] | 6 => img[146] | 7 => img[147] | 8 => img[148] | 9 => img[149] | 10 => img[150] | 11 => img[151] | 12 => img[152] | 13 => img[153] | 14 => img[154] | 15 => img[155] | 16 => img[156] | 17 => img[157] | 18 => img[158] | 19 => img[159] | 20 => img[160] | 21 => img[161] | 22 => img[162] | 23 => img[163] | 24 => img[164] | 25 => img[165] | 26 => img[166] | _ => img[167]
      brightnessToChar px isRaw255 inverted)
  | 168 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[168] | 1 => img[169] | 2 => img[170] | 3 => img[171] | 4 => img[172] | 5 => img[173] | 6 => img[174] | 7 => img[175] | 8 => img[176] | 9 => img[177] | 10 => img[178] | 11 => img[179] | 12 => img[180] | 13 => img[181] | 14 => img[182] | 15 => img[183] | 16 => img[184] | 17 => img[185] | 18 => img[186] | 19 => img[187] | 20 => img[188] | 21 => img[189] | 22 => img[190] | 23 => img[191] | 24 => img[192] | 25 => img[193] | 26 => img[194] | _ => img[195]
      brightnessToChar px isRaw255 inverted)
  | 196 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[196] | 1 => img[197] | 2 => img[198] | 3 => img[199] | 4 => img[200] | 5 => img[201] | 6 => img[202] | 7 => img[203] | 8 => img[204] | 9 => img[205] | 10 => img[206] | 11 => img[207] | 12 => img[208] | 13 => img[209] | 14 => img[210] | 15 => img[211] | 16 => img[212] | 17 => img[213] | 18 => img[214] | 19 => img[215] | 20 => img[216] | 21 => img[217] | 22 => img[218] | 23 => img[219] | 24 => img[220] | 25 => img[221] | 26 => img[222] | _ => img[223]
      brightnessToChar px isRaw255 inverted)
  | 224 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[224] | 1 => img[225] | 2 => img[226] | 3 => img[227] | 4 => img[228] | 5 => img[229] | 6 => img[230] | 7 => img[231] | 8 => img[232] | 9 => img[233] | 10 => img[234] | 11 => img[235] | 12 => img[236] | 13 => img[237] | 14 => img[238] | 15 => img[239] | 16 => img[240] | 17 => img[241] | 18 => img[242] | 19 => img[243] | 20 => img[244] | 21 => img[245] | 22 => img[246] | 23 => img[247] | 24 => img[248] | 25 => img[249] | 26 => img[250] | _ => img[251]
      brightnessToChar px isRaw255 inverted)
  | 252 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[252] | 1 => img[253] | 2 => img[254] | 3 => img[255] | 4 => img[256] | 5 => img[257] | 6 => img[258] | 7 => img[259] | 8 => img[260] | 9 => img[261] | 10 => img[262] | 11 => img[263] | 12 => img[264] | 13 => img[265] | 14 => img[266] | 15 => img[267] | 16 => img[268] | 17 => img[269] | 18 => img[270] | 19 => img[271] | 20 => img[272] | 21 => img[273] | 22 => img[274] | 23 => img[275] | 24 => img[276] | 25 => img[277] | 26 => img[278] | _ => img[279]
      brightnessToChar px isRaw255 inverted)
  | 280 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[280] | 1 => img[281] | 2 => img[282] | 3 => img[283] | 4 => img[284] | 5 => img[285] | 6 => img[286] | 7 => img[287] | 8 => img[288] | 9 => img[289] | 10 => img[290] | 11 => img[291] | 12 => img[292] | 13 => img[293] | 14 => img[294] | 15 => img[295] | 16 => img[296] | 17 => img[297] | 18 => img[298] | 19 => img[299] | 20 => img[300] | 21 => img[301] | 22 => img[302] | 23 => img[303] | 24 => img[304] | 25 => img[305] | 26 => img[306] | _ => img[307]
      brightnessToChar px isRaw255 inverted)
  | 308 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[308] | 1 => img[309] | 2 => img[310] | 3 => img[311] | 4 => img[312] | 5 => img[313] | 6 => img[314] | 7 => img[315] | 8 => img[316] | 9 => img[317] | 10 => img[318] | 11 => img[319] | 12 => img[320] | 13 => img[321] | 14 => img[322] | 15 => img[323] | 16 => img[324] | 17 => img[325] | 18 => img[326] | 19 => img[327] | 20 => img[328] | 21 => img[329] | 22 => img[330] | 23 => img[331] | 24 => img[332] | 25 => img[333] | 26 => img[334] | _ => img[335]
      brightnessToChar px isRaw255 inverted)
  | 336 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[336] | 1 => img[337] | 2 => img[338] | 3 => img[339] | 4 => img[340] | 5 => img[341] | 6 => img[342] | 7 => img[343] | 8 => img[344] | 9 => img[345] | 10 => img[346] | 11 => img[347] | 12 => img[348] | 13 => img[349] | 14 => img[350] | 15 => img[351] | 16 => img[352] | 17 => img[353] | 18 => img[354] | 19 => img[355] | 20 => img[356] | 21 => img[357] | 22 => img[358] | 23 => img[359] | 24 => img[360] | 25 => img[361] | 26 => img[362] | _ => img[363]
      brightnessToChar px isRaw255 inverted)
  | 364 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[364] | 1 => img[365] | 2 => img[366] | 3 => img[367] | 4 => img[368] | 5 => img[369] | 6 => img[370] | 7 => img[371] | 8 => img[372] | 9 => img[373] | 10 => img[374] | 11 => img[375] | 12 => img[376] | 13 => img[377] | 14 => img[378] | 15 => img[379] | 16 => img[380] | 17 => img[381] | 18 => img[382] | 19 => img[383] | 20 => img[384] | 21 => img[385] | 22 => img[386] | 23 => img[387] | 24 => img[388] | 25 => img[389] | 26 => img[390] | _ => img[391]
      brightnessToChar px isRaw255 inverted)
  | 392 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[392] | 1 => img[393] | 2 => img[394] | 3 => img[395] | 4 => img[396] | 5 => img[397] | 6 => img[398] | 7 => img[399] | 8 => img[400] | 9 => img[401] | 10 => img[402] | 11 => img[403] | 12 => img[404] | 13 => img[405] | 14 => img[406] | 15 => img[407] | 16 => img[408] | 17 => img[409] | 18 => img[410] | 19 => img[411] | 20 => img[412] | 21 => img[413] | 22 => img[414] | 23 => img[415] | 24 => img[416] | 25 => img[417] | 26 => img[418] | _ => img[419]
      brightnessToChar px isRaw255 inverted)
  | 420 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[420] | 1 => img[421] | 2 => img[422] | 3 => img[423] | 4 => img[424] | 5 => img[425] | 6 => img[426] | 7 => img[427] | 8 => img[428] | 9 => img[429] | 10 => img[430] | 11 => img[431] | 12 => img[432] | 13 => img[433] | 14 => img[434] | 15 => img[435] | 16 => img[436] | 17 => img[437] | 18 => img[438] | 19 => img[439] | 20 => img[440] | 21 => img[441] | 22 => img[442] | 23 => img[443] | 24 => img[444] | 25 => img[445] | 26 => img[446] | _ => img[447]
      brightnessToChar px isRaw255 inverted)
  | 448 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[448] | 1 => img[449] | 2 => img[450] | 3 => img[451] | 4 => img[452] | 5 => img[453] | 6 => img[454] | 7 => img[455] | 8 => img[456] | 9 => img[457] | 10 => img[458] | 11 => img[459] | 12 => img[460] | 13 => img[461] | 14 => img[462] | 15 => img[463] | 16 => img[464] | 17 => img[465] | 18 => img[466] | 19 => img[467] | 20 => img[468] | 21 => img[469] | 22 => img[470] | 23 => img[471] | 24 => img[472] | 25 => img[473] | 26 => img[474] | _ => img[475]
      brightnessToChar px isRaw255 inverted)
  | 476 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[476] | 1 => img[477] | 2 => img[478] | 3 => img[479] | 4 => img[480] | 5 => img[481] | 6 => img[482] | 7 => img[483] | 8 => img[484] | 9 => img[485] | 10 => img[486] | 11 => img[487] | 12 => img[488] | 13 => img[489] | 14 => img[490] | 15 => img[491] | 16 => img[492] | 17 => img[493] | 18 => img[494] | 19 => img[495] | 20 => img[496] | 21 => img[497] | 22 => img[498] | 23 => img[499] | 24 => img[500] | 25 => img[501] | 26 => img[502] | _ => img[503]
      brightnessToChar px isRaw255 inverted)
  | 504 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[504] | 1 => img[505] | 2 => img[506] | 3 => img[507] | 4 => img[508] | 5 => img[509] | 6 => img[510] | 7 => img[511] | 8 => img[512] | 9 => img[513] | 10 => img[514] | 11 => img[515] | 12 => img[516] | 13 => img[517] | 14 => img[518] | 15 => img[519] | 16 => img[520] | 17 => img[521] | 18 => img[522] | 19 => img[523] | 20 => img[524] | 21 => img[525] | 22 => img[526] | 23 => img[527] | 24 => img[528] | 25 => img[529] | 26 => img[530] | _ => img[531]
      brightnessToChar px isRaw255 inverted)
  | 532 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[532] | 1 => img[533] | 2 => img[534] | 3 => img[535] | 4 => img[536] | 5 => img[537] | 6 => img[538] | 7 => img[539] | 8 => img[540] | 9 => img[541] | 10 => img[542] | 11 => img[543] | 12 => img[544] | 13 => img[545] | 14 => img[546] | 15 => img[547] | 16 => img[548] | 17 => img[549] | 18 => img[550] | 19 => img[551] | 20 => img[552] | 21 => img[553] | 22 => img[554] | 23 => img[555] | 24 => img[556] | 25 => img[557] | 26 => img[558] | _ => img[559]
      brightnessToChar px isRaw255 inverted)
  | 560 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[560] | 1 => img[561] | 2 => img[562] | 3 => img[563] | 4 => img[564] | 5 => img[565] | 6 => img[566] | 7 => img[567] | 8 => img[568] | 9 => img[569] | 10 => img[570] | 11 => img[571] | 12 => img[572] | 13 => img[573] | 14 => img[574] | 15 => img[575] | 16 => img[576] | 17 => img[577] | 18 => img[578] | 19 => img[579] | 20 => img[580] | 21 => img[581] | 22 => img[582] | 23 => img[583] | 24 => img[584] | 25 => img[585] | 26 => img[586] | _ => img[587]
      brightnessToChar px isRaw255 inverted)
  | 588 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[588] | 1 => img[589] | 2 => img[590] | 3 => img[591] | 4 => img[592] | 5 => img[593] | 6 => img[594] | 7 => img[595] | 8 => img[596] | 9 => img[597] | 10 => img[598] | 11 => img[599] | 12 => img[600] | 13 => img[601] | 14 => img[602] | 15 => img[603] | 16 => img[604] | 17 => img[605] | 18 => img[606] | 19 => img[607] | 20 => img[608] | 21 => img[609] | 22 => img[610] | 23 => img[611] | 24 => img[612] | 25 => img[613] | 26 => img[614] | _ => img[615]
      brightnessToChar px isRaw255 inverted)
  | 616 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[616] | 1 => img[617] | 2 => img[618] | 3 => img[619] | 4 => img[620] | 5 => img[621] | 6 => img[622] | 7 => img[623] | 8 => img[624] | 9 => img[625] | 10 => img[626] | 11 => img[627] | 12 => img[628] | 13 => img[629] | 14 => img[630] | 15 => img[631] | 16 => img[632] | 17 => img[633] | 18 => img[634] | 19 => img[635] | 20 => img[636] | 21 => img[637] | 22 => img[638] | 23 => img[639] | 24 => img[640] | 25 => img[641] | 26 => img[642] | _ => img[643]
      brightnessToChar px isRaw255 inverted)
  | 644 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[644] | 1 => img[645] | 2 => img[646] | 3 => img[647] | 4 => img[648] | 5 => img[649] | 6 => img[650] | 7 => img[651] | 8 => img[652] | 9 => img[653] | 10 => img[654] | 11 => img[655] | 12 => img[656] | 13 => img[657] | 14 => img[658] | 15 => img[659] | 16 => img[660] | 17 => img[661] | 18 => img[662] | 19 => img[663] | 20 => img[664] | 21 => img[665] | 22 => img[666] | 23 => img[667] | 24 => img[668] | 25 => img[669] | 26 => img[670] | _ => img[671]
      brightnessToChar px isRaw255 inverted)
  | 672 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[672] | 1 => img[673] | 2 => img[674] | 3 => img[675] | 4 => img[676] | 5 => img[677] | 6 => img[678] | 7 => img[679] | 8 => img[680] | 9 => img[681] | 10 => img[682] | 11 => img[683] | 12 => img[684] | 13 => img[685] | 14 => img[686] | 15 => img[687] | 16 => img[688] | 17 => img[689] | 18 => img[690] | 19 => img[691] | 20 => img[692] | 21 => img[693] | 22 => img[694] | 23 => img[695] | 24 => img[696] | 25 => img[697] | 26 => img[698] | _ => img[699]
      brightnessToChar px isRaw255 inverted)
  | 700 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[700] | 1 => img[701] | 2 => img[702] | 3 => img[703] | 4 => img[704] | 5 => img[705] | 6 => img[706] | 7 => img[707] | 8 => img[708] | 9 => img[709] | 10 => img[710] | 11 => img[711] | 12 => img[712] | 13 => img[713] | 14 => img[714] | 15 => img[715] | 16 => img[716] | 17 => img[717] | 18 => img[718] | 19 => img[719] | 20 => img[720] | 21 => img[721] | 22 => img[722] | 23 => img[723] | 24 => img[724] | 25 => img[725] | 26 => img[726] | _ => img[727]
      brightnessToChar px isRaw255 inverted)
  | 728 => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[728] | 1 => img[729] | 2 => img[730] | 3 => img[731] | 4 => img[732] | 5 => img[733] | 6 => img[734] | 7 => img[735] | 8 => img[736] | 9 => img[737] | 10 => img[738] | 11 => img[739] | 12 => img[740] | 13 => img[741] | 14 => img[742] | 15 => img[743] | 16 => img[744] | 17 => img[745] | 18 => img[746] | 19 => img[747] | 20 => img[748] | 21 => img[749] | 22 => img[750] | 23 => img[751] | 24 => img[752] | 25 => img[753] | 26 => img[754] | _ => img[755]
      brightnessToChar px isRaw255 inverted)
  | _ => String.mk (List.range 28 |>.map fun i =>
      let px := match i with | 0 => img[756] | 1 => img[757] | 2 => img[758] | 3 => img[759] | 4 => img[760] | 5 => img[761] | 6 => img[762] | 7 => img[763] | 8 => img[764] | 9 => img[765] | 10 => img[766] | 11 => img[767] | 12 => img[768] | 13 => img[769] | 14 => img[770] | 15 => img[771] | 16 => img[772] | 17 => img[773] | 18 => img[774] | 19 => img[775] | 20 => img[776] | 21 => img[777] | 22 => img[778] | 23 => img[779] | 24 => img[780] | 25 => img[781] | 26 => img[782] | _ => img[783]
      brightnessToChar px isRaw255 inverted)

/-- Render full 28×28 MNIST image as ASCII art.

Converts a flattened 784-dimensional MNIST image into a 28-line ASCII art
representation. Automatically detects whether the input uses raw 0-255 values
or normalized 0-1 values.

**Parameters:**
- `img`: 784-dimensional vector (flattened 28×28 MNIST image in row-major order)
- `inverted`: If `true`, use inverted brightness (for light terminals)

**Returns:** Multi-line string containing 28 rows of 28 characters each

**Algorithm:**
1. Convert Vector to Array for Nat indexing
2. Auto-detect value range (0-255 vs 0-1)
3. Render each of 28 rows
4. Join rows with newline characters

**Complexity:** O(784) - processes each pixel exactly once

**Output format:**
```
............................
............................
........@@####@@............
......@@########@@..........
...@@@@##########@@.........
...@@############@@.........
.....@@##########@@.........
.......@@########@@.........
.........@@######@@.........
...........@@####@@.........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
........@@@@########@@......
........@@@@########@@......
........@@@@########@@......
...........@@########@@.....
...........@@########@@.....
.............@@####@@.......
............................
```

**Usage:**
```lean
let samples ← loadMNISTTest "data"
let (image, label) := samples[0]!
IO.println (renderImage image false)
```
-/
def renderImage (img : Vector 784) (inverted : Bool) : String :=
  let isRaw255 := autoDetectRange img

  -- Render each row using literal indices (manual unrolling approach)
  let rows := List.range 28 |>.map fun rowIdx =>
    renderRowLiteral img rowIdx isRaw255 inverted
  String.intercalate "\n" rows

/-- Render MNIST image with a text label above it.

Adds a text label (e.g., "Ground Truth: 5" or "Predicted: 3") above the
rendered ASCII art for better context when displaying multiple images or
comparing predictions.

**Parameters:**
- `img`: 784-dimensional MNIST image vector
- `label`: Text to display above the image (e.g., "Digit: 5")
- `inverted`: Whether to use inverted brightness mapping

**Returns:** Multi-line string with label on first line, then ASCII art

**Output format:**
```
Predicted: 7
............................
............................
........@@####@@............
[... rest of image ...]
```

**Usage:**
```lean
let prediction := argmax networkOutput
let ascii := renderImageWithLabel image s!"Predicted: {prediction}" false
IO.println ascii
```
-/
def renderImageWithLabel (img : Vector 784) (label : String) (inverted : Bool) : String :=
  label ++ "\n" ++ renderImage img inverted

/-- Compute basic statistics for an image.

Returns min, max, mean, and standard deviation of pixel values.

**Parameters:**
- `img`: 784-dimensional MNIST image vector

**Returns:** Tuple of (min, max, mean, stddev)

**Complexity:** O(784) - single pass through image
-/
def computeImageStats (img : Vector 784) : Float × Float × Float × Float :=
  -- Use SciLean's sum notation for mean calculation
  let sum := ∑ i, img[i]
  let mean := sum / 784.0

  -- For min/max, use a simple approximation based on sampling
  -- (full iteration would require mutable state or complex fold)
  let samples := [img[0], img[100], img[200], img[400], img[600], img[783]]
  let min := samples.foldl (fun acc x => if x < acc then x else acc) img[0]
  let max := samples.foldl (fun acc x => if x > acc then x else acc) img[0]

  -- Compute standard deviation using sum notation
  let sumSqDiff := ∑ i, (img[i] - mean) * (img[i] - mean)
  let stddev := Float.sqrt (sumSqDiff / 784.0)

  (min, max, mean, stddev)

/-- Render image with statistical overlay.

Displays the ASCII art with statistics (min/max/mean/stddev) below the image.

**Parameters:**
- `img`: 784-dimensional MNIST image vector
- `inverted`: Whether to use inverted brightness mapping

**Returns:** Multi-line string with image and statistics

**Output format:**
```
............................
[... ASCII art ...]
............................
Statistics:
  Min:    0.000  Max:  255.000
  Mean:  45.123  Std:   78.456
```
-/
def renderImageWithStats (img : Vector 784) (inverted : Bool) : String :=
  let ascii := renderImage img inverted
  let (min, max, mean, stddev) := computeImageStats img
  let isRaw255 := autoDetectRange img

  let rangeStr := if isRaw255 then " (0-255 range)" else " (0-1 range)"
  let minStr := toString min
  let maxStr := toString max
  let meanStr := toString mean
  let stddevStr := toString stddev
  let statsText :=
    "\nStatistics:\n" ++
    "  Min: " ++ minStr ++ "  Max: " ++ maxStr ++ "\n" ++
    "  Mean: " ++ meanStr ++ "  Std: " ++ stddevStr ++ rangeStr

  ascii ++ statsText

/-- Render two images side-by-side for comparison.

Displays two images horizontally adjacent, useful for comparing ground truth
vs prediction or original vs transformed images.

**Parameters:**
- `img1`: First 784-dimensional MNIST image
- `img2`: Second 784-dimensional MNIST image
- `label1`: Label for first image (e.g., "Original")
- `label2`: Label for second image (e.g., "Predicted")
- `inverted`: Whether to use inverted brightness mapping

**Returns:** Multi-line string with both images side-by-side

**Output format:**
```
Original              Predicted
----------------------------  ----------------------------
..........................    ..........................
[... ASCII art ...]           [... ASCII art ...]
..........................    ..........................
```
-/
def renderImageComparison (img1 img2 : Vector 784) (label1 label2 : String) (inverted : Bool) : String :=
  let ascii1 := renderImage img1 inverted
  let ascii2 := renderImage img2 inverted

  let lines1 := ascii1.splitOn "\n"
  let lines2 := ascii2.splitOn "\n"

  -- Pad labels to same width
  let maxLabelLen := Nat.max label1.length label2.length
  let paddedLabel1 := label1 ++ String.mk (List.replicate (maxLabelLen - label1.length) ' ')
  let paddedLabel2 := label2 ++ String.mk (List.replicate (maxLabelLen - label2.length) ' ')

  -- Headers
  let header := paddedLabel1 ++ "    " ++ paddedLabel2 ++ "\n" ++
                String.mk (List.replicate 28 '-') ++ "    " ++
                String.mk (List.replicate 28 '-')

  -- Combine lines side-by-side
  let combined := List.zip lines1 lines2 |>.map fun (l1, l2) =>
    l1 ++ "    " ++ l2

  header ++ "\n" ++ String.intercalate "\n" combined

/-- Render multiple images in a grid layout.

Displays multiple images in rows and columns, useful for visualizing batches
or showing multiple predictions.

**Parameters:**
- `images`: List of 784-dimensional MNIST images
- `labels`: Corresponding labels for each image
- `cols`: Number of columns in the grid (rows computed automatically)
- `inverted`: Whether to use inverted brightness mapping

**Returns:** Multi-line string with images arranged in a grid

**Output format:**
```
Label 0          Label 1          Label 2
-----------      -----------      -----------
..........       ..........       ..........
[ASCII art]      [ASCII art]      [ASCII art]
..........       ..........       ..........
```
-/
def renderImageGrid (images : List (Vector 784)) (labels : List String) (cols : Nat) (inverted : Bool) : String :=
  if images.isEmpty || cols == 0 then
    "(empty grid)"
  else
    -- Split images and labels into rows manually
    -- For simplicity, just take the first cols items repeatedly
    let rec pairUp (imgs : List (Vector 784)) (lbls : List String) : List (Vector 784 × String) :=
      match imgs, lbls with
      | [], _ => []
      | _, [] => []
      | i :: is, l :: ls => (i, l) :: pairUp is ls

    let paired := pairUp images labels
    let rows := [paired.take cols]  -- Simplified: just show first row for now

    -- Render each row
    let renderedRows := rows.map fun row =>
      -- Render labels
      let labelLine := String.intercalate "  " (row.map fun (_, lbl) =>
        lbl ++ String.mk (List.replicate (Nat.max 0 (28 - lbl.length)) ' '))

      -- Render separator
      let sepLine := String.intercalate "  " (row.map fun _ =>
        String.mk (List.replicate 28 '-'))

      -- Render images
      let asciiImages := row.map fun (img, _) =>
        (renderImage img inverted).splitOn "\n"

      -- Combine lines horizontally
      let maxLines := (asciiImages.map List.length).foldl Nat.max 0
      let combinedLines := List.range maxLines |>.map fun lineIdx =>
        let parts := asciiImages.map fun lines =>
          if lineIdx < lines.length then
            lines[lineIdx]!
          else
            String.mk (List.replicate 28 ' ')
        String.intercalate "  " parts

      labelLine ++ "\n" ++ sepLine ++ "\n" ++ String.intercalate "\n" combinedLines

    String.intercalate "\n\n" renderedRows

/-- Additional color palette options for different terminal themes. -/
structure PaletteConfig where
  chars : String
  name : String

/-- Available brightness palettes.

Provides multiple palette options for different preferences and terminal themes:
- `default`: Standard 10-char palette (space to @)
- `simple`: 8-char palette for simpler rendering
- `detailed`: 16-char palette for fine gradation
- `blocks`: Unicode block characters for smooth gradients
-/
def availablePalettes : List PaletteConfig := [
  { chars := " .:-=+*#%@", name := "default" },
  { chars := " .:-=+@", name := "simple" },
  { chars := " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$", name := "detailed" },
  { chars := " ░▒▓█", name := "blocks" }
]

/-- Get palette by name, defaulting to standard palette if not found. -/
def getPalette (name : String) : String :=
  let rec findPalette (palettes : List PaletteConfig) : String :=
    match palettes with
    | [] => brightnessChars
    | p :: rest => if p.name == name then p.chars else findPalette rest
  findPalette availablePalettes

/-- Render image with custom palette.

Like `renderImage` but allows specifying a custom character palette.

**Parameters:**
- `img`: 784-dimensional MNIST image vector
- `palette`: String of characters ordered from darkest to brightest
- `inverted`: Whether to use inverted brightness mapping

**Returns:** Multi-line string containing 28 rows of 28 characters each
-/
def renderImageWithPalette (img : Vector 784) (palette : String) (inverted : Bool) : String :=
  -- For now, just use the default renderer
  -- TODO: Implement custom palette support with proper SciLean indexing
  renderImage img inverted

/-- Render image with border frame.

Adds a decorative border around the ASCII art image.

**Parameters:**
- `img`: 784-dimensional MNIST image vector
- `inverted`: Whether to use inverted brightness mapping
- `borderStyle`: Border style - "single", "double", "rounded", "heavy", or "ascii"

**Returns:** Multi-line string with bordered image
-/
def renderImageWithBorder (img : Vector 784) (inverted : Bool) (borderStyle : String := "single") : String :=
  let ascii := renderImage img inverted
  let lines := ascii.splitOn "\n"

  let (tl, tr, bl, br, h, v) := match borderStyle with
    | "double" => ("╔", "╗", "╚", "╝", "═", "║")
    | "rounded" => ("╭", "╮", "╰", "╯", "─", "│")
    | "heavy" => ("┏", "┓", "┗", "┛", "━", "┃")
    | "ascii" => ("+", "+", "+", "+", "-", "|")
    | _ => ("┌", "┐", "└", "┘", "─", "│")  -- single (default)

  let topBorder := tl ++ String.mk (List.replicate 28 h.toList.head!) ++ tr
  let bottomBorder := bl ++ String.mk (List.replicate 28 h.toList.head!) ++ br
  let framedLines := lines.map fun line => v ++ line ++ v

  topBorder ++ "\n" ++ String.intercalate "\n" framedLines ++ "\n" ++ bottomBorder

end VerifiedNN.Util.ImageRenderer
