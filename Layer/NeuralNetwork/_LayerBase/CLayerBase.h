//============================================
// 全てのニューラルネットワーク系レイヤーのベースとなる共通処理
// 自前で実装可能であれば必ず継承する必要はない
//============================================
#ifndef __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_H__
#define __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_H__


#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>
#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>
#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	template<class IOLayer>
	class CLayerBase : public IOLayer
	{
	protected:
		enum ProcessType
		{
			PROCESSTYPE_NONE,
			PROCESSTYPE_CALCULATE,
			PROCESSTYPE_LEARN,

			PROCESSTYPE_COUNT
		};

	private:
		Gravisbell::GUID guid;	/**< レイヤー識別用のGUID */
		SettingData::Standard::IData* pRuntimeParameter;

		ProcessType processType;
		U32 batchSize;	/**< バッチサイズ */

	public:
		/** コンストラクタ */
		CLayerBase(Gravisbell::GUID guid, SettingData::Standard::IData* i_pRuntimeParameter)
			:	guid				(guid)
			,	pRuntimeParameter	(i_pRuntimeParameter)
			,	processType			(PROCESSTYPE_CALCULATE)
		{
		}
		/** デストラクタ */
		virtual ~CLayerBase()
		{
			if(pRuntimeParameter)
				delete pRuntimeParameter;
		}


		//====================================
		// 実行時設定
		//====================================
	public:
		/** 実行時設定を取得する. */
		const SettingData::Standard::IData* GetRuntimeParameter()const
		{
			return pRuntimeParameter;
		}
		SettingData::Standard::IData* GetRuntimeParameter()
		{
			return pRuntimeParameter;
		}

		/** 学習設定のアイテムを取得する.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID. */
		SettingData::Standard::IItemBase* GetRuntimeParameterItem(const wchar_t* i_dataID)
		{
			// 学習設定データを取得
			Gravisbell::SettingData::Standard::IData* pLearnSettingData = this->GetRuntimeParameter();
			if(pLearnSettingData == NULL)
				return NULL;

			// 該当IDの設定アイテムを取得
			Gravisbell::SettingData::Standard::IItemBase* pItem = pLearnSettingData->GetItemByID(i_dataID);
			if(pItem == NULL)
				return NULL;

			return pItem;
		}

		/** 実行時設定を設定する.
			int型、float型、enum型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
		{
			// 該当IDの設定アイテムを取得
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_INT:
				{
					Gravisbell::SettingData::Standard::IItem_Int* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue(i_param);
				}
				return ErrorCode::ERROR_CODE_NONE;

			case Gravisbell::SettingData::Standard::ITEMTYPE_FLOAT:
				{
					Gravisbell::SettingData::Standard::IItem_Float* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue((F32)i_param);
				}
				return ErrorCode::ERROR_CODE_NONE;

			case Gravisbell::SettingData::Standard::ITEMTYPE_ENUM:
				{
					Gravisbell::SettingData::Standard::IItem_Enum* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Enum*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue(i_param);
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** 実行時設定を設定する.
			int型、float型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
		{
			// 該当IDの設定アイテムを取得
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_INT:
				{
					Gravisbell::SettingData::Standard::IItem_Int* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue((S32)i_param);
				}
				return ErrorCode::ERROR_CODE_NONE;

			case Gravisbell::SettingData::Standard::ITEMTYPE_FLOAT:
				{
					Gravisbell::SettingData::Standard::IItem_Float* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue((F32)i_param);
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** 実行時設定を設定する.
			bool型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
		{
			// 該当IDの設定アイテムを取得
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_BOOL:
				{
					Gravisbell::SettingData::Standard::IItem_Bool* pItemBool = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Bool*>(pItem);
					if(pItemBool == NULL)
						break;
					pItemBool->SetValue(i_param);

				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** 実行時設定を設定する.
			string型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
		{
			// 該当IDの設定アイテムを取得
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_STRING:
				{
					Gravisbell::SettingData::Standard::IItem_String* pItemString = dynamic_cast<Gravisbell::SettingData::Standard::IItem_String*>(pItem);
					if(pItemString == NULL)
						break;
					pItemString->SetValue(i_param);

				}
				return ErrorCode::ERROR_CODE_NONE;
			case Gravisbell::SettingData::Standard::ITEMTYPE_ENUM:
				{
					Gravisbell::SettingData::Standard::IItem_Enum* pItemString = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Enum*>(pItem);
					if(pItemString == NULL)
						break;
					pItemString->SetValue(i_param);
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}


		//===========================
		// レイヤー共通
		//===========================

		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const
		{
			return this->guid;
		}

		/** レイヤーの種類識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const
		{
			return this->GetLayerData().GetLayerCode();
		}

		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const
		{
			return this->GetLayerData().GetLayerStructure();
		}

		//===========================
		// レイヤー処理関連
		//===========================
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			this->batchSize = batchSize;
			this->processType = PROCESSTYPE_LEARN;

			return PreProcessLearn();
		}
		virtual ErrorCode PreProcessLearn() = 0;

		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			this->batchSize = batchSize;
			this->processType = PROCESSTYPE_CALCULATE;

			return PreProcessCalculate();
		}
		virtual ErrorCode PreProcessCalculate() = 0;


		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		U32 GetBatchSize()const
		{
			return this->batchSize;
		}

		/** 演算種別を取得する */
		ProcessType GetProcessType()const
		{
			return this->processType;
		}




		//===========================
		// レイヤーデータ関連
		//===========================
	public:
		/** レイヤーデータを取得する */
		virtual ILayerData& GetLayerData() = 0;
		virtual const ILayerData& GetLayerData()const = 0;
	};

	class CNNSingle2SingleLayerBase : public CLayerBase<INNSingle2SingleLayer>
	{
	private:
		IODataStruct inputDataStruct;	/**< 入力データ構造 */
		IODataStruct outputDataStruct;	/**< 出力データ構造 */

	public:
		/** コンストラクタ */
		CNNSingle2SingleLayerBase(Gravisbell::GUID guid, SettingData::Standard::IData* i_pRuntimeParameter, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBase			(guid, i_pRuntimeParameter)
			,	inputDataStruct		(i_inputDataStruct)
			,	outputDataStruct	(i_outputDataStruct)
		{
		}
		/** デストラクタ */
		virtual ~CNNSingle2SingleLayerBase()
		{
		}


	public:
		//===========================
		// レイヤー共通
		//===========================
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
		}

	public:
		//===========================
		// 入力レイヤー関連
		//===========================
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const
		{
			return this->inputDataStruct;
		}

		/** 入力バッファ数を取得する. */
		unsigned int GetInputBufferCount()const
		{
			return this->GetInputDataStruct().GetDataCount();
		}


	public:
		//===========================
		// 出力レイヤー関連
		//===========================
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** 出力バッファ数を取得する */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};

	class CNNSingle2MultLayerBase : public CLayerBase<INNSingle2MultLayer>
	{
	private:
		IODataStruct inputDataStruct;	/**< 入力データ構造 */
		IODataStruct outputDataStruct;	/**< 出力データ構造 */

	public:
		/** コンストラクタ */
		CNNSingle2MultLayerBase(Gravisbell::GUID guid, SettingData::Standard::IData* i_pRuntimeParameter, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBase			(guid, i_pRuntimeParameter)
			,	inputDataStruct		(i_inputDataStruct)
			,	outputDataStruct	(i_outputDataStruct)
		{
		}
		/** デストラクタ */
		virtual ~CNNSingle2MultLayerBase()
		{
		}


	public:
		//===========================
		// レイヤー共通
		//===========================
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_OUTPUT;
		}

	public:
		//===========================
		// 入力レイヤー関連
		//===========================
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const
		{
			return this->inputDataStruct;
		}

		/** 入力バッファ数を取得する. */
		unsigned int GetInputBufferCount()const
		{
			return this->GetInputDataStruct().GetDataCount();
		}


	public:
		//===========================
		// 出力レイヤー関連
		//===========================
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** 出力バッファ数を取得する */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};

	class CNNMult2SingleLayerBase : public CLayerBase<INNMult2SingleLayer>
	{
	private:
		std::vector<IODataStruct>	lpInputDataStruct;	/**< 入力データ構造 */
		IODataStruct				outputDataStruct;	/**< 出力データ構造 */

	public:
		/** コンストラクタ */
		CNNMult2SingleLayerBase(Gravisbell::GUID guid, SettingData::Standard::IData* i_pRuntimeParameter, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBase			(guid, i_pRuntimeParameter)
			,	lpInputDataStruct	(i_lpInputDataStruct)
			,	outputDataStruct	(i_outputDataStruct)
		{
		}
		/** デストラクタ */
		virtual ~CNNMult2SingleLayerBase()
		{
		}


	public:
		//===========================
		// レイヤー共通
		//===========================
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
		}

	public:
		//===========================
		// 入力レイヤー関連
		//===========================
		/** 入力データの数を取得する */
		U32 GetInputDataCount()const
		{
			return (U32)this->lpInputDataStruct.size();
		}

		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct(U32 i_dataNum)const
		{
			if(i_dataNum >= this->GetInputDataCount())
				return IODataStruct(0,0,0,0);

			return this->lpInputDataStruct[i_dataNum];
		}

		/** 入力バッファ数を取得する. */
		U32 GetInputBufferCount(U32 i_dataNum)const
		{
			return this->GetInputDataStruct(i_dataNum).GetDataCount();
		}



	public:
		//===========================
		// 出力レイヤー関連
		//===========================
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** 出力バッファ数を取得する */
		U32 GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
