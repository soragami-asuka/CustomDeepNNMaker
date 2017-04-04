//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
//======================================
#include<Layer/NeuralNetwork/INeuralNetwork.h>


#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>

#include"LayerConnect.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_Base : public INeuralNetwork
	{
	private:
		std::map<Gravisbell::GUID, ILayerConnect*>	lpLayerInfo;	/**< �S���C���[�̊Ǘ��N���X. <���C���[GUID, ���C���[�ڑ����̃A�h���X> */

		std::list<ILayerConnect*> lpCalculateLayer0List;		/**< ���C���[���������ɕ��ׂ����X�g.  */

		ILayerConnect* pOutputLayer;	/**< �o�͐M���ɐݒ肳��Ă��郌�C���[�̃A�h���X. */

	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_Base();
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_Base();

	public:
		//====================================
		// ���C���[�̒ǉ�
		//====================================
		/** ���C���[��ǉ�����.
			�ǉ��������C���[�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
			@param pLayer	�ǉ����郌�C���[�̃A�h���X. */
		ErrorCode AddLayer(INNLayer* pLayer);
		/** ���C���[���폜����.
			@param i_guid	�폜���郌�C���[��GUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** ���C���[��S�폜���� */
		ErrorCode EraseAllLayer();

		/** �o�^����Ă��郌�C���[�����擾���� */
		ErrorCode GetLayerCount()const;
		/** ���C���[��ԍ��w��Ŏ擾���� */
		const INNLayer* GetLayerByNum(const U32 i_layerNum);
		/** ���C���[��GUID�w��Ŏ擾���� */
		const INNLayer* GetLayerByGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// ���o�̓��C���[
		//====================================
		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		GUID GetInputGUID()const;

		/** �o�͐M���Ɋ��蓖�Ă�Ă��郌�C���[��GUID���擾���� */
		GUID GetOutputLayerGUID()const;
		/** �o�͐M�����C���[��ݒ肷�� */
		GUID SetOutputLayerGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// ���C���[�̐ڑ�
		//====================================
		/** ���C���[�ɓ��̓��C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID);
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID);

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetInputLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetBypassLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;


		/** ���C���[�̐ڑ���ԂɈُ킪�Ȃ����`�F�b�N����.
			@param	o_errorLayer	�G���[�������������C���[GUID�i�[��. 
			@return	�ڑ��Ɉُ킪�Ȃ��ꍇ��NO_ERROR, �ُ킪�������ꍇ�ُ͈���e��Ԃ��A�Ώۃ��C���[��GUID��o_errorLayer�Ɋi�[����. */
		ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer);



		//====================================
		// �w�K�ݒ�
		//====================================
		/** �w�K�ݒ���擾����.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			@param	guid	�擾�Ώۃ��C���[��GUID. */
		const SettingData::Standard::IData GetLearnSettingData(const Gravisbell::GUID& guid);

		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			@param	guid	�擾�Ώۃ��C���[��GUID
			@param	data	�ݒ肷��w�K�ݒ� */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& data);

		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param);
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param);
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			bool�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param);
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			string�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param);

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell