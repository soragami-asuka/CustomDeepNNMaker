//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#ifndef __FullyConnect_DATA_BASE_H__
#define __FullyConnect_DATA_BASE_H__

#include<Layer/ILayerData.h>
#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Layer/NeuralNetwork/IWeightData.h>

#include<vector>

#include"FullyConnect_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class FullyConnect_LayerData_Base : public ILayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤーデータ識別用のGUID */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
		FullyConnect::LayerStructure layerStructure;	/**< レイヤー構造 */

		IWeightData* pWeightData;	/**< 重み情報 */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		FullyConnect_LayerData_Base(const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~FullyConnect_LayerData_Base();


		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		virtual ErrorCode Initialize(void) = 0;
		/** 初期化. 各ニューロンの値をランダムに初期化
			@param	i_config			設定情報
			@oaram	i_inputDataStruct	入力データ構造情報
			@return	成功した場合0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data);
		/** 初期化. バッファからデータを読み込む
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	成功した場合0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize);

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
		U64 GetUseBufferByteCount()const;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const;

	public:
		//===========================
		// レイヤー構造
		//===========================
		/** 入力データ構造が使用可能か確認する.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@return	使用可能な入力データ構造の場合trueが返る. */
		bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** 出力データ構造を取得する.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
		IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** 複数出力が可能かを確認する */
		bool CheckCanHaveMultOutputLayer(void);


		//===========================
		// 固有関数
		//===========================
	public:
		/** 入力バッファ数を取得する */
		U32 GetInputBufferCount()const;

		/** ニューロン数を取得する */
		U32 GetNeuronCount()const;


		//===========================
		// オプティマイザー設定
		//===========================
	public:
		/** オプティマイザーを変更する */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]);
		/** オプティマイザーのハイパーパラメータを変更する */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif