//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#ifndef __FullyConnect_LAYERDATA_CPU_H__
#define __FullyConnect_LAYERDATA_CPU_H__

#include"FullyConnect_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FullyConnect_LayerData_CPU : public FullyConnect_LayerData_Base
	{
		friend class FullyConnect_CPU;

	private:
		// 本体
		std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< 各ニューロンの係数<ニューロン数, 入力数> */
		std::vector<NEURON_TYPE>				lpBias;				/**< ニューロンのバイアス<ニューロン数> */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		FullyConnect_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~FullyConnect_LayerData_CPU();


		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);
		/** 初期化. 各ニューロンの値をランダムに初期化
			@param	i_config			設定情報
			@oaram	i_inputDataStruct	入力データ構造情報
			@return	成功した場合0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct);
		/** 初期化. バッファからデータを読み込む
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	成功した場合0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize);


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		INNLayer* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif