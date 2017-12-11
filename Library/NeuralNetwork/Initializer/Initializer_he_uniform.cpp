//=====================================
// パラメータ初期化クラス.
// Heの一様分布. limit  = sqrt(6 / fan_in)の一様分布
//=====================================
#include"stdafx.h"

#include"Initializer_he_uniform.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_he_uniform::Initializer_he_uniform(Random& random)
	:	random	(random)
{
}
/** デストラクタ1 */
Initializer_he_uniform::~Initializer_he_uniform()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_he_uniform::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	F32 limit = sqrtf(6.0f / i_outputCount);

	return random.GetUniformValue(-limit, +limit);
}

