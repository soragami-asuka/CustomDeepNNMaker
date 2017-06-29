//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __Residual_DATA_BASE_H__
#define __Residual_DATA_BASE_H__

#include<Layer/ILayerData.h>

#include<vector>

#include"Residual_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Residual_LayerData_Base : public ILayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤーデータ識別用のGUID */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
//		Residual::LayerStructure layerStructure;	/**< レイヤー構造 */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Residual_LayerData_Base(const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~Residual_LayerData_Base();


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
		ErrorCode Initialize(const SettingData::Standard::IData& i_data);
		/** 初期化. バッファからデータを読み込む
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	成功した場合0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize );


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


		//===========================
		// オプティマイザー設定
		//===========================		
	public:
		/** オプティマイザーを変更する */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]){return ErrorCode::ERROR_CODE_NONE;}
		/** オプティマイザーのハイパーパラメータを変更する */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value){return ErrorCode::ERROR_CODE_NONE;}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value){return ErrorCode::ERROR_CODE_NONE;}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]){return ErrorCode::ERROR_CODE_NONE;}
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif