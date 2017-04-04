//=======================================
// �j���[�����l�b�g���[�N�{�̒�`
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INeuralNetwork : public INNLayer
	{
	public:
		/** �R���X�g���N�^ */
		INeuralNetwork(){}
		/** �f�X�g���N�^ */
		virtual ~INeuralNetwork(){}

	public:
		//====================================
		// ���C���[�̒ǉ�
		//====================================
		/** ���C���[��ǉ�����.
			�ǉ��������C���[�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
			@param pLayer	�ǉ����郌�C���[�̃A�h���X. */
		virtual ErrorCode AddLayer(INNLayer* pLayer) = 0;
		/** ���C���[���폜����.
			@param i_guid	�폜���郌�C���[��GUID */
		virtual ErrorCode EraseLayer(const Gravisbell::GUID& i_guid) = 0;
		/** ���C���[��S�폜���� */
		virtual ErrorCode EraseAllLayer() = 0;

		/** �o�^����Ă��郌�C���[�����擾���� */
		virtual ErrorCode GetLayerCount()const = 0;
		/** ���C���[��ԍ��w��Ŏ擾���� */
		virtual const INNLayer* GetLayerByNum(const U32 i_layerNum) = 0;
		/** ���C���[��GUID�w��Ŏ擾���� */
		virtual const INNLayer* GetLayerByGUID(const Gravisbell::GUID& i_guid) = 0;


		//====================================
		// ���o�̓��C���[
		//====================================
		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		virtual GUID GetInputGUID()const = 0;

		/** �o�͐M���Ɋ��蓖�Ă�Ă��郌�C���[��GUID���擾���� */
		virtual GUID GetOutputLayerGUID()const = 0;
		/** �o�͐M�����C���[��ݒ肷�� */
		virtual GUID SetOutputLayerGUID(const Gravisbell::GUID& i_guid) = 0;


		//====================================
		// ���C���[�̐ڑ�
		//====================================
		/** ���C���[�ɓ��̓��C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		virtual ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		virtual ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID) = 0;
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID) = 0;

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const = 0;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		virtual ErrorCode GetInputLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const = 0;

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const = 0;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		virtual ErrorCode GetBypassLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const = 0;


		/** ���C���[�̐ڑ���ԂɈُ킪�Ȃ����`�F�b�N����.
			@param	o_errorLayer	�G���[�������������C���[GUID�i�[��. 
			@return	�ڑ��Ɉُ킪�Ȃ��ꍇ��NO_ERROR, �ُ킪�������ꍇ�ُ͈���e��Ԃ��A�Ώۃ��C���[��GUID��o_errorLayer�Ɋi�[����. */
		virtual ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer) = 0;



		//====================================
		// �w�K�ݒ�
		//====================================
		/** �w�K�ݒ���擾����.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			@param	guid	�擾�Ώۃ��C���[��GUID. */
		virtual const SettingData::Standard::IData* GetLearnSettingData(const Gravisbell::GUID& guid) = 0;

		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			@param	guid	�擾�Ώۃ��C���[��GUID
			@param	data	�ݒ肷��w�K�ݒ� */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& data) = 0;

		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			bool�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			string�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param) = 0;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
