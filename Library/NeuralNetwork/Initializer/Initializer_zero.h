//=====================================
// パラメータ初期化クラス.
// 全て0で初期化
//=====================================
#ifndef __GRAVISBELL_NN_INITIALIZER_ZERO_H__
#define __GRAVISBELL_NN_INITIALIZER_ZERO_H__

#include"Initializer_base.h"
#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Initializer_zero : public Initializer_base
	{
	public:
		/** コンストラクタ */
		Initializer_zero();
		/** デストラクタ1 */
		virtual ~Initializer_zero();


	public:
		//===========================
		// パラメータの値を取得
		//===========================
		/** パラメータの値を取得する.
			@param	i_inputCount	入力信号数.
			@param	i_outputCount	出力信号数. */
		F32 GetParameter(U32 i_inputCount, U32 i_outputCount);
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif	__GRAVISBELL_NN_INITIALIZER_ZERO_H__