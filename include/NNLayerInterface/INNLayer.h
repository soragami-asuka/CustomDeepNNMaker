//=======================================
// レイヤーベース
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_BASE_H__
#define __GRAVISBELL_I_NN_LAYER_BASE_H__

#include"ISingleInputLayer.h"
#include"IOutputLayer.h"

#include"../SettingData/Standard/IData.h"


namespace Gravisbell {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNLayer : public ISingleInputLayer, public virtual IOutputLayer
	{
	public:
		/** コンストラクタ */
		INNLayer(){}
		/** デストラクタ */
		virtual ~INNLayer(){}

	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		virtual ErrorCode Initialize(void) = 0;
		/** 初期化. 各ニューロンの値をランダムに初期化
			@param	i_config			設定情報
			@oaram	i_inputDataStruct	入力データ構造情報
			@return	成功した場合0 */
		virtual ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct) = 0;
		/** 初期化. バッファからデータを読み込む
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	成功した場合0 */
		virtual ErrorCode InitializeFromBuffer(BYTE* i_lpBuffer, int i_bufferSize) = 0;

		/** レイヤーの設定情報を取得する */
		virtual const SettingData::Standard::IData* GetLayerConfig()const = 0;

		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;

	public:
		/** 演算処理を実行する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

		/** 学習差分をレイヤーに反映させる.
			入力信号、出力信号は直前のCalculateの値を参照する.
			出力誤差差分、入力誤差差分は直前のCalculateLearnErrorの値を参照する. */
		virtual ErrorCode ReflectionLearnError(void) = 0;
	};


}	// NeuralNetwork
}	// Gravisbell

#endif