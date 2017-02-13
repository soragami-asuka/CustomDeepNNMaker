//==================================
// �ݒ荀��(�񋓌^)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"
#include"INNLayerConfigItemEx_Enum.h"
#include"NNLayerConfigItemBase.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Enum : public NNLayerConfigItemBase<INNLayerConfigItemEx_Enum>
	{
	private:
		struct EnumItem
		{
			std::string id;
			std::string name;
			std::string comment;

			/** �R���X�g���N�^ */
			EnumItem()
				: id ("")
				, name ("")
				, comment ("")
			{
			}
			/** �R���X�g���N�^ */
			EnumItem(const std::string& id, const std::string& name, const std::string& comment)
				: id		(id)
				, name		(name)
				, comment	(comment)
			{
			}
			/** �R�s�[�R���X�g���N�^ */
			EnumItem(const EnumItem& item)
				: id		(item.id)
				, name		(item.name)
				, comment	(item.comment)
			{
			}

			/** =���Z */
			const EnumItem& operator=(const EnumItem& item)
			{
				this->id = item.id;
				this->name = item.name;
				this->comment = item.comment;

				return *this;
			}


			bool operator==(const EnumItem& item)const
			{
				if(this->id != item.id)
					return false;
				if(this->name != item.name)
					return false;
				if(this->name != item.comment)
					return false;
				return true;
			}
			bool operator!=(const EnumItem& item)const
			{
				return !(*this == item);
			}
		};

	private:
		std::vector<EnumItem> lpEnumItem;

		std::string value;
		std::string defaultValue;


	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItem_Enum(const char i_szID[], const char i_szName[], const char i_szText[])
			: NNLayerConfigItemBase(i_szID, i_szName, i_szText)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		NNLayerConfigItem_Enum(const NNLayerConfigItem_Enum& item)
			: NNLayerConfigItemBase(item)
			, lpEnumItem (item.lpEnumItem)
			, value (item.value)
			, defaultValue (item.defaultValue)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~NNLayerConfigItem_Enum(){}
		
		/** ��v���Z */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const NNLayerConfigItem_Enum* pItem = dynamic_cast<const NNLayerConfigItem_Enum*>(&item);
			if(pItem == NULL)
				return false;

			if(NNLayerConfigItemBase::operator!=(*pItem))
				return false;

			for(unsigned int itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum] != pItem->lpEnumItem[itemNum])
					return false;
			}
			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const INNLayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** ���g�̕������쐬���� */
		INNLayerConfigItemBase* Clone()const
		{
			return new NNLayerConfigItem_Enum(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_ENUM;
		}

	public:
		/** �l���擾���� */
		int GetValue()const
		{
			return this->GetNumByID(this->value.c_str());
		}

		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		ELayerErrorCode SetValue(int value)
		{
			if(value < 0)
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;
			if(value >= (int)this->lpEnumItem.size())
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

			this->value = this->lpEnumItem[value].id;

			return LAYER_ERROR_NONE;
		}
		/** �l��ݒ肷��(������w��)
			@param i_szID	�ݒ肷��l(������w��)
			@return ���������ꍇ0 */
		ELayerErrorCode SetValue(const char i_szEnumID[])
		{
			return this->SetValue(this->GetNumByID(i_szEnumID));
		}

	public:
		/** �񋓗v�f�����擾���� */
		unsigned int GetEnumCount()const
		{
			return this->lpEnumItem.size();
		}
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		int GetEnumID(int num, char o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;
			
			memcpy(o_szBuf, this->lpEnumItem[num].id.c_str(), this->lpEnumItem[num].id.size() + 1);

			return 0;
		}
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		int GetEnumName(int num, char o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;
			
			memcpy(o_szBuf, this->lpEnumItem[num].name.c_str(), this->lpEnumItem[num].name.size() + 1);

			return 0;
		}
		/** �񋓗v�f�R�����g��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		int GetEnumComment(int num, char o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;
			
			memcpy(o_szBuf, this->lpEnumItem[num].comment.c_str(), this->lpEnumItem[num].comment.size() + 1);

			return 0;
		}

		/** ID���w�肵�ė񋓗v�f�ԍ����擾����
			@param i_szBuf�@���ׂ��ID.
			@return ���������ꍇ0<=num<GetEnumCount�̒l. ���s�����ꍇ�͕��̒l���Ԃ�. */
		int GetNumByID(const char i_szEnumID[])const
		{
			std::string enumID = i_szEnumID;

			for(unsigned int itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum].id == enumID)
					return (int)itemNum;
			}

			return -1;
		}

		/** �f�t�H���g�̐ݒ�l���擾���� */
		int GetDefault()const
		{
			return this->GetNumByID(this->defaultValue.c_str());
		}

	public:
		/** �񋓒l��ǉ�����.
			����ID�����ɑ��݂���ꍇ���s����.
			@param szEnumID		�ǉ�����ID
			@param szEnumName	�ǉ����閼�O
			@param szComment	�ǉ�����R�����g��.
			@return ���������ꍇ�A�ǉ����ꂽ�A�C�e���̔ԍ����Ԃ�. ���s�����ꍇ�͕��̒l���Ԃ�. */
		int AddEnumItem(const char szEnumID[], const char szEnumName[], const char szComment[])
		{
			// ����ID�����݂��邩�m�F
			int sameID = this->GetNumByID(szEnumID);
			if(sameID >= 0)
				return -1;

			std::string id = szEnumID;
			if(id.size()+1 >= ID_BUFFER_MAX)
				return -1;

			// �ǉ�
			this->lpEnumItem.push_back(EnumItem(id, szEnumName, szComment));

			return this->lpEnumItem.size()-1;
		}

		/** �񋓒l���폜����.
			@param num	�폜����񋓔ԍ�
			@return ���������ꍇ0 */
		int EraseEnumItem(int num)
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;

			// iterator��i�߂�
			auto it = this->lpEnumItem.begin();
			for(int i=0; i<num; i++)
				it++;

			// �폜
			this->lpEnumItem.erase(it);

			return 0;
		}
		/** �񋓒l���폜����
			@param szEnumID �폜�����ID
			@return ���������ꍇ0 */
		int EraseEnumItem(const char szEnumID[])
		{
			return this->EraseEnumItem(this->GetNumByID(szEnumID));
		}

		/** �f�t�H���g�l��ݒ肷��.	�ԍ��w��.
			@param num �f�t�H���g�l�ɐݒ肷��ԍ�. 
			@return ���������ꍇ0 */
		int SetDefaultItem(int num)
		{
			char szEnumID[ID_BUFFER_MAX];

			// ID���擾����
			if(this->GetEnumID(num, szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}
		/** �f�t�H���g�l��ݒ肷��.	ID�w��. 
			@param szID �f�t�H���g�l�ɐݒ肷��ID. 
			@return ���������ꍇ0 */
		int SetDefaultItem(const char szEnumID[])
		{
			if(this->GetNumByID(szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}


	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

			byteCount += sizeof(unsigned int);		// �l�̃o�b�t�@�T�C�Y
			byteCount += this->value.size();		// �l

			return byteCount;
		}

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			unsigned int bufferPos = 0;

			// �l
			{
				// �o�b�t�@�T�C�Y
				unsigned int bufferSize = *(unsigned int*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(unsigned int);

				// �l
				std::vector<char> szBuf(bufferSize+1, NULL);
				for(unsigned int i=0; i<bufferSize; i++)
				{
					szBuf[i] = i_lpBuffer[bufferPos++];
				}
				std::string value = &szBuf[0];


				this->SetValue(value.c_str());
			}


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// �l
			{
				// �o�b�t�@�T�C�Y
				unsigned int bufferSize = this->value.size();;
				*(unsigned int*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(unsigned int);

				// �l
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize);
				bufferPos += bufferSize;
			}


			return bufferPos;
		}
	};
	
	/** �ݒ荀��(�񋓒l)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItemEx_Enum* CreateLayerCofigItem_Enum(const char i_szID[], const char i_szName[], const char i_szText[])
	{
		return new NNLayerConfigItem_Enum(i_szID, i_szName, i_szText);
	}
}