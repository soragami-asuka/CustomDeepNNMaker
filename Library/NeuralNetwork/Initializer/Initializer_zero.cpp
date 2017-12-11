//=====================================
// パラメータ初期化クラス.
// 全て0で初期化
//=====================================
#include"stdafx.h"

#include"Initializer_zero.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_zero::Initializer_zero()
{
}
/** デストラクタ1 */
Initializer_zero::~Initializer_zero()
{
}


//===========================
// パラメータの値を取得
//===========================
/** パラメータの値を取得する.
	@param	i_inputCount	入力信号数.
	@param	i_outputCount	出力信号数. */
F32 Initializer_zero::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return 0.0f;
}

