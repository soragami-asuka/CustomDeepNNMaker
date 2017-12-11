//=====================================
// パラメータ初期化クラス.
// average=0.0 variance=1.0の正規乱数で初期化
//=====================================
#include"stdafx.h"

#include"Initializer_normal.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_normal::Initializer_normal(Random& random)
	:	random	(random)
{
}
/** デストラクタ1 */
Initializer_normal::~Initializer_normal()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_normal::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetNormalValue(0.0f, 1.0f);
}

