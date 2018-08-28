//=======================================
// ニューラルネットワーク本体定義
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNMult2SingleLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INeuralNetwork : public INNMult2SingleLayer
	{
	public:
		/** コンストラクタ */
		INeuralNetwork(){}
		/** デストラクタ */
		virtual ~INeuralNetwork(){}

	public:
		//====================================
		// 学習設定
		//====================================
		/** 学習設定を取得する.
			@param	guid	取得対象レイヤーのGUID. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid)const = 0;

		/** 学習設定を設定する.
			int型、float型、enum型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param) = 0;
		/** 学習設定を設定する.
			int型、float型が対象.
			@param	guid		取得対象レイヤーのGUID. 指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param) = 0;
		/** 学習設定を設定する.
			bool型が対象.
			@param	guid		取得対象レイヤーのGUID. 指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param) = 0;
		/** 学習設定を設定する.
			string型が対象.
			@param	guid		取得対象レイヤーのGUID. 指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param) = 0;

	public:
		//==========================================
		// 演算処理.
		// 入出力はCPU側のメモリー
		//==========================================
		/** 演算処理を実行する.
			@param i_lppInputBuffer		入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]) = 0;

		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

	public:
		//==========================================
		// 出力バッファの取得
		//==========================================		
		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		virtual ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
