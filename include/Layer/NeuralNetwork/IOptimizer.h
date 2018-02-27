//========================================
// パラメータ更新のための最適化ルーチン
//========================================
#ifndef __GRAVISBELL_I_NN_OPTIMIZER_H__
#define __GRAVISBELL_I_NN_OPTIMIZER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 最適化ルーチン */
	class IOptimizer
	{
	public:
		//===========================
		// コンストラクタ/デストラクタ
		//===========================
		/** コンストラクタ */
		IOptimizer(){}
		/** デストラクタ */
		virtual ~IOptimizer(){}

	public:
		//===========================
		// 基本情報
		//===========================
		/** 識別IDの取得 */
		virtual const wchar_t* GetOptimizerID()const = 0;

		/** ハイパーパラメータを設定する
			@param	i_parameterID	パラメータ識別用ID
			@param	i_value			パラメータ. */
		virtual ErrorCode SetHyperParameter(const wchar_t i_parameterID[], F32 i_value) = 0;
		/** ハイパーパラメータを設定する
			@param	i_parameterID	パラメータ識別用ID
			@param	i_value			パラメータ. */
		virtual ErrorCode SetHyperParameter(const wchar_t i_parameterID[], S32 i_value) = 0;
		/** ハイパーパラメータを設定する
			@param	i_parameterID	パラメータ識別用ID
			@param	i_value			パラメータ. */
		virtual ErrorCode SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]) = 0;

	public:
		//===========================
		// 処理
		//===========================
		/** パラメータを更新する.
			@param io_lpParamter	更新するパラメータ.
			@param io_lpDParameter	パラメータの変化量. */
		virtual ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[]) = 0;

	public:
		//===========================
		// 保存
		//===========================
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		virtual U64 GetUseBufferByteCount()const = 0;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
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
