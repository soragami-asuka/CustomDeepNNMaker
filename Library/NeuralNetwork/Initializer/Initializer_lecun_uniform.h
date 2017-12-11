//=====================================
// パラメータ初期化クラス.
// LeCunの一様分布. limit = sqrt(3 / fan_in)の一様分布
//=====================================
#ifndef __GRAVISBELL_NN_INITIALIZER_LECUN_UNIFORM_H__
#define __GRAVISBELL_NN_INITIALIZER_LECUN_UNIFORM_H__

#include"Initializer_base.h"
#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Initializer_lecun_uniform : public Initializer_base
	{
	private:
		Random& random;

	public:
		/** コンストラクタ */
		Initializer_lecun_uniform(Random& random);
		/** デストラクタ1 */
		virtual ~Initializer_lecun_uniform();


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