//======================================
// 活性関数のレイヤーデータ
//======================================
#ifndef __ACTIVATION_DATA_BASE_H__
#define __ACTIVATION_DATA_BASE_H__

#include<Layer/NeuralNetwork/INNLayerData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"Activation_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Activation_LayerData_Base : public INNLayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤーデータ識別用のGUID */

		IODataStruct inputDataStruct;	/**< 入力データ構造 */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
		Activation::LayerStructure layerStructure;	/**< レイヤー構造 */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Activation_LayerData_Base(const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~Activation_LayerData_Base();


		//===========================
		// 共通処理
		//===========================
	public:
		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const;

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
		// レイヤー設定
		//===========================
	public:
		/** 設定情報を設定 */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		U32 GetUseBufferByteCount()const;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;

		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. */
		U32 GetInputBufferCount()const;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する */
		unsigned int GetOutputBufferCount()const;
		

		//===========================
		// 固有関数
		//===========================
	public:
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif