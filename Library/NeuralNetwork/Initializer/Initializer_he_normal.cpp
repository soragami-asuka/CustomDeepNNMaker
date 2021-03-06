//=====================================
// パラメータ初期化クラス.
// Heの正規分布. stddev = sqrt(2 / fan_in)の切断正規分布
//=====================================
#include"stdafx.h"

#include"Initializer_he_normal.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_he_normal::Initializer_he_normal(Random& random)
	:	random	(random)
{
}
/** デストラクタ1 */
Initializer_he_normal::~Initializer_he_normal()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_he_normal::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetTruncatedNormalValue(0.0f, sqrtf(2.0f / i_inputCount) );
}

