//======================================
// バッチ正規化のレイヤーデータ
//======================================
#ifndef __BatchNormalization_DATA_BASE_H__
#define __BatchNormalization_DATA_BASE_H__

#include<Layer/IO/ISingleInputLayerData.h>
#include<Layer/IO/ISingleOutputLayerData.h>
#include<Layer/NeuralNetwork/INNLayer.h>
#include<Layer/NeuralNetwork/IOptimizer.h>

#include<vector>

#include"BatchNormalization_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class BatchNormalization_LayerData_Base : public IO::ISingleInputLayerData, public IO::ISingleOutputLayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤーデータ識別用のGUID */

		IODataStruct inputDataStruct;	/**< 入力データ構造 */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
		BatchNormalization::LayerStructure layerStructure;	/**< レイヤー構造 */

		IOptimizer* m_pOptimizer_scale;		/**< スケール更新用オプティマイザ */
		IOptimizer* m_pOptimizer_bias;		/**< バイアス更新用オプティマイザ */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		BatchNormalization_LayerData_Base(const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~BatchNormalization_LayerData_Base();


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
		U32 GetUseBufferByteCount()const;


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


		//===========================
		// オプティマイザー設定
		//===========================		
	public:
		/** オプティマイザーを変更する */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** オプティマイザーのハイパーパラメータを変更する */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif