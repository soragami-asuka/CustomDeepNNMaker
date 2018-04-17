//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#include"stdafx.h"

#include"SOM_LayerData_Base.h"
#include"SOM_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	SOM_LayerData_Base::SOM_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< レイヤー構造を定義したコンフィグクラス */
		,	layerStructure	()		/**< レイヤー構造 */
	{
	}
	/** デストラクタ */
	SOM_LayerData_Base::~SOM_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;

	}


	//===========================
	// 共通処理
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID SOM_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID SOM_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode SOM_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
	{
		Gravisbell::ErrorCode err = ERROR_CODE_NONE;

		// レイヤーコードを確認
		{
			Gravisbell::GUID config_guid;
			err = config.GetLayerCode(config_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			Gravisbell::GUID layer_guid;
			err = ::GetLayerCode(layer_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			if(config_guid != layer_guid)
				return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
		}

		if(this->pLayerStructure != NULL)
			delete this->pLayerStructure;
		this->pLayerStructure = config.Clone();

		// 構造体に読み込む
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* SOM_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U64 SOM_LayerData_Base::GetUseBufferByteCount()const
	{
		U64 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;


		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// 本体のバイト数
		bufferSize += (this->GetUnitCount() * this->layerStructure.InputBufferCount) * sizeof(F32);	// ニューロン係数
		bufferSize += this->GetUnitCount() * sizeof(F32);	// バイアス係数

		return bufferSize;
	}


	//===========================
	// レイヤー構造
	//===========================
	/** 入力データ構造が使用可能か確認する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	使用可能な入力データ構造の場合trueが返る. */
	bool SOM_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;

		if(i_lpInputDataStruct[0].GetDataCount() != this->layerStructure.InputBufferCount)
			return false;

		return true;
	}

	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct SOM_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);

		return IODataStruct(this->layerStructure.DimensionCount, 1, 1, 1);
	}

	/** 複数出力が可能かを確認する */
	bool SOM_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}


	
	//===========================
	// 固有関数
	//===========================
	/** 入力バッファ数を取得する */
	U32 SOM_LayerData_Base::GetInputBufferCount()const
	{
		return this->layerStructure.InputBufferCount;
	}
	/** ユニット数を取得する */
	U32 SOM_LayerData_Base::GetUnitCount()const
	{
		return (U32)pow(this->layerStructure.ResolutionCount, this->layerStructure.DimensionCount);
	}


	//===========================
	// オプティマイザー設定
	//===========================
	/** オプティマイザーを変更する */
	ErrorCode SOM_LayerData_Base::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		return ERROR_CODE_NONE;
	}
	/** オプティマイザーのハイパーパラメータを変更する */
	ErrorCode SOM_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode SOM_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode SOM_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//==================================
	// SOM関連処理
	//==================================
	/** マップサイズを取得する.
		@return	マップのバッファ数を返す. */
	U32 SOM_LayerData_Base::GetMapSize()const
	{
		return this->GetUnitCount() * this->GetInputBufferCount();
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
