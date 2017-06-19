//========================================
// 最適化ルーチン
//========================================
#ifndef __GRAVISBELL_I_NN_OPTIMIZER_H__
#define __GRAVISBELL_I_NN_OPTIMIZER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 最適化ルーチン種別 */
	enum OptimizerType
	{
		OPTIMIZER_TYPE_SGD,			// SGD
		OPTIMIZER_TYPE_MOMENTUM,	// Momentum
		OPTIMIZER_TYPE_ADADGRAD,	// AdaGrad
		OPTIMIZER_TYPE_RMSPROP,		// RMSprop
		OPTIMIZER_TYPE_ADADELTA,	// AdaDelta
		OPTIMIZER_TYPE_ADAM,		// Adam

		OPTIMIZER_TYPE_COUNT
	};

	/** 最適化ルーチン */
	class IOptimizer
	{
	public:
		/** コンストラクタ */
		IOptimizer(){}
		/** デストラクタ */
		virtual ~IOptimizer(){}

	public:
		//===========================
		// 基本情報
		//===========================
		/** オプティマイザの種別を取得する */
		virtual OptimizerType GetTypeCode()const = 0;

	public:
		//===========================
		// 処理
		//===========================
		/** パラメータを更新する.
			@param io_lpParamter	更新するパラメータ.
			@param io_lpDParameter	パラメータの変化量. */
		virtual ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[]) = 0;
	};

	/** SGD */
	class iOptimizer_SGD : public IOptimizer
	{
	public:
		/** コンストラクタ */
		iOptimizer_SGD(){}
		/** デストラクタ */
		virtual ~iOptimizer_SGD(){}

	public:
		//===========================
		// 基本情報
		//===========================
		/** ハイパーパラメータを更新する
			@param	i_learnCoeff	学習係数 */
		virtual ErrorCode SetHyperParameter(F32 i_learnCoeff) = 0;
	};

	/** Momentum */
	class iOptimizer_Momentum : public IOptimizer
	{
	public:
		/** コンストラクタ */
		iOptimizer_Momentum(){}
		/** デストラクタ */
		virtual ~iOptimizer_Momentum(){}

	public:
		//===========================
		// 基本情報
		//===========================
		/** ハイパーパラメータを更新する
			@param	i_learnCoeff	学習係数
			@param	i_alpha			慣性. */
		virtual ErrorCode SetHyperParameter(F32 i_learnCoeff, F32 i_alpha) = 0;
	};

	/** AdaDelta */
	class iOptimizer_AdaDelta : public IOptimizer
	{
	public:
		/** コンストラクタ */
		iOptimizer_AdaDelta(){}
		/** デストラクタ */
		virtual ~iOptimizer_AdaDelta(){}

	public:
		//===========================
		// 基本情報
		//===========================
		/** ハイパーパラメータを更新する
			@param	i_rho			減衰率.
			@param	i_epsilon		補助係数. */
		virtual ErrorCode SetHyperParameter(F32 i_rho, F32 i_epsilon) = 0;
	};

	/** Adam */
	class iOptimizer_Adam : public IOptimizer
	{
	public:
		/** コンストラクタ */
		iOptimizer_Adam(){}
		/** デストラクタ */
		virtual ~iOptimizer_Adam(){}

	public:
		//===========================
		// 基本情報
		//===========================
		/** ハイパーパラメータを更新する
			@param	i_learnCoeff	学習係数
			@param	i_alpha			慣性.
			@param	i_beta1			減衰率.
			@param	i_beta2			減衰率.
			@param	i_epsilon		補助係数. */
		virtual ErrorCode SetHyperParameter(F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon) = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell



#endif	__GRAVISBELL_I_NN_OPTIMIZER_H__
