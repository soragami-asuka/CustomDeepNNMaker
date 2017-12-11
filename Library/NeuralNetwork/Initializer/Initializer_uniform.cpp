//=====================================
// パラメータ初期化クラス.
// -1～+1の一様乱数で初期化
//=====================================
#include"stdafx.h"

#include"Initializer_uniform.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_uniform::Initializer_uniform(Random& random)
	:	random	(random)
{
}
/** デストラクタ1 */
Initializer_uniform::~Initializer_uniform()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_uniform::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetUniformValue(-1.0f, +1.0f);
}

