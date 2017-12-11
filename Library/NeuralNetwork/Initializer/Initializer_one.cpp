//=====================================
// パラメータ初期化クラス.
// 全て1で初期化
//=====================================
#include"stdafx.h"

#include"Initializer_one.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_one::Initializer_one()
{
}
/** デストラクタ1 */
Initializer_one::~Initializer_one()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_one::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return 1.0f;
}

