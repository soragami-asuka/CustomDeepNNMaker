//======================================
// バッチ正規化のレイヤーデータ
//======================================
#ifndef __ExponentialNormalization_DATA_BASE_H__
#define __ExponentialNormalization_DATA_BASE_H__

#include<Layer/ILayerData.h>
#include<Layer/NeuralNetwork/IOptimizer.h>

#include<vector>

#include"ExponentialNormalization_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class ExponentialNormalization_LayerData_Base : public ILayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤーデータ識別用のGUID */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
		ExponentialNormalization::LayerStructure layerStructure;	/**< レイヤー構造 */

		U64 learnTime;	/**< 学習回数 */

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		ExponentialNormalization_LayerData_Base(const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~ExponentialNormalization_LayerData_Base();


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