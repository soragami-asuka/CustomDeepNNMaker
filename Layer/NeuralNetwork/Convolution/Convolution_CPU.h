//======================================
// 畳み込みニューラルネットワークの結合処理レイヤー
// CPU処理用
//======================================
#ifndef __CONVOLUTION_CPU_H__
#define __CONVOLUTION_CPU_H__

#include"stdafx.h"

#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Convolution_CPU : public Convolution_Base
{
private:
	// データ本体
	class Convolution_LayerData_CPU& layerData;

	// 入出力バッファ
	std::vector<std::vector<F32>>			lpOutputBuffer;		/**< 出力バッファ <バッチ数><畳み込み数> */
	std::vector<std::vector<F32>>			lpDInputBuffer;		/**< 入力誤差差分 <バッチ数><入力信号数> */

	std::vector<F32*>						lppBatchOutputBuffer;		/**< バッチ処理用出力バッファ <バッチ数> */
	std::vector<F32*>						lppBatchDInputBuffer;		/**< バッチ処理用入力誤差差分 <バッチ数> */

	// Get関数を使うと処理不可がかさむので一時保存用. PreCalculateで値を格納.
	U32 filterSize;						/**< フィルタサイズ */
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 neuronCount;					/**< ニューロン数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */

	IODataStruct paddingInputDataStruct;	/**< パディング後の入力バッファの入力データ構造 */
	std::vector<std::vector<F32>> lpPaddingInputBuffer;	/**< パディング後の入力バッファ <バッチ数><入力バッファ> */

	// 演算時の入力データ
	CONST_BATCH_BUFFER_POINTER m_lppInputBuffer;		/**< 演算時の入力データ */
	CONST_BATCH_BUFFER_POINTER m_lppDOutputBuffer;		/**< 入力誤差計算時の出力誤差データ */

	// 演算処理用のバッファ
	bool onUseDropOut;											/**< ドロップアウト処理を実行するフラグ. */
	std::vector<std::vector<NEURON_TYPE>>	lppDropOutBuffer;	/**< ドロップアウト処理用の係数<ニューロン数, フィルタサイズ> */


public:
	/** コンストラクタ */
	Convolution_CPU(Gravisbell::GUID guid, class Convolution_LayerData_CPU& i_layerData);
	/** デストラクタ */
	virtual ~Convolution_CPU();


public:
	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 GetLayerKind()const;

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Initialize(void);


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Convolution_LayerData_Base& GetLayerData();
	const Convolution_LayerData_Base& GetLayerData()const;


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 WriteToBuffer(BYTE* o_lpBuffer)const;


public:
	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode PreProcessLearn(unsigned int batchSize);

	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessCalculate(unsigned int batchSize);


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessCalculateLoop();


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);

	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

public:
	//================================
	// 学習処理
	//================================
	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** 学習差分をレイヤーに反映させる.
		入力信号、出力信号は直前のCalculateの値を参照する.
		出力誤差差分、入力誤差差分は直前のCalculateLearnErrorの値を参照する. */
	ErrorCode ReflectionLearnError(void);

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif