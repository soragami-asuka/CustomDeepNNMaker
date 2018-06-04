//========================================
// パラメータ初期化ルーチン
//========================================
#ifndef __GRAVISBELL_I_NN_WEIGHTDATA_H__
#define __GRAVISBELL_I_NN_WEIGHTDATA_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"
#include"../../Common/IODataStruct.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 初期化ルーチン */
	class IWeightData
	{
	public:
		//===========================
		// コンストラクタ/デストラクタ
		//===========================
		/** コンストラクタ */
		IWeightData(){}
		/** デストラクタ */
		virtual ~IWeightData(){}

	public:
		//===========================
		// 初期化
		//===========================
		virtual ErrorCode Initialize(const wchar_t i_initializerID[], U32 i_inputCount, U32 i_outputCount) = 0;

		virtual S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize) = 0;


		//===========================
		// サイズを取得
		//===========================
		/** Weightのサイズを取得する */
		virtual U64 GetWeigthSize()const = 0;
		/** Biasのサイズを取得する */
		virtual U64 GetBiasSize()const = 0;


		//===========================
		// 値を取得
		//===========================
		/** Weightを取得する */
		virtual const F32* GetWeight()const = 0;
		/** Biasを取得する */
		virtual const F32* GetBias()const = 0;


		//===========================
		// 値を更新
		//===========================
		/** Weigth,Biasを設定する.
			@param	lpWeight	設定するWeightの値.
			@param	lpBias		設定するBiasの値. */
		virtual ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias) = 0;
		/** Weight,Biasを更新する.
			@param	lpDWeight	Weightの変化量.
			@param	lpDBias		Biasのh変化量. */
		virtual ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias) = 0;


		//===========================
		// オプティマイザー設定
		//===========================
		/** オプティマイザーを変更する */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** オプティマイザーのハイパーパラメータを変更する */
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]) = 0;


		//===========================
		// レイヤー保存
		//===========================
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		virtual U64 GetUseBufferByteCount()const = 0;
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell



#endif	__GRAVISBELL_I_NN_INITIALIZER_H__
