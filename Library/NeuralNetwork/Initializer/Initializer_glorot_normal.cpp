//=====================================
// パラメータ初期化クラス.
// Glorotの正規分布. stddev = sqrt(2 / (fan_in + fan_out))の切断正規分布
//=====================================
#include"stdafx.h"

#include"Initializer_glorot_normal.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_glorot_normal::Initializer_glorot_normal(Random& random)
	:	random	(random)
{
}
/** デストラクタ1 */
Initializer_glorot_normal::~Initializer_glorot_normal()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_glorot_normal::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetTruncatedNormalValue(0.0f, sqrtf(2.0f / (i_inputCount + i_outputCount)) );
}

